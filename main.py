import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# --- 0. TELEGRAM AYARLARI ---
# GitHub'ın kasasından (Secrets) şifreleri alır
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("Telegram ayarları eksik!")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Telegram Hatası: {e}")

# --- 1. HASTA LİSTESİ (BIST 100) ---
def get_bist100_tickers():
    return [
        "AEFES.IS", "AGHOL.IS", "AHGAZ.IS", "AKBNK.IS", "AKCNS.IS", "AKFGY.IS", "AKSA.IS", "AKSEN.IS", "ALARK.IS", 
        "ALBRK.IS", "ALFAS.IS", "ARCLK.IS", "ASELS.IS", "ASTOR.IS", "ASUZU.IS", "AYDEM.IS", "AYGAZ.IS", "BAGFS.IS", 
        "BERA.IS", "BIMAS.IS", "BIOEN.IS", "BRSAN.IS", "BRYAT.IS", "BUCIM.IS", "CANTE.IS", "CCOLA.IS", "CEMTS.IS", 
        "CIMSA.IS", "CWENE.IS", "DOAS.IS", "DOHOL.IS", "ECILC.IS", "ECZYT.IS", "EGEEN.IS", "EKGYO.IS", "ENERY.IS", 
        "ENJSA.IS", "ENKAI.IS", "EREGL.IS", "EUPWR.IS", "EUREN.IS", "FROTO.IS", "GARAN.IS", "GENIL.IS", "GESAN.IS", 
        "GLYHO.IS", "GSDHO.IS", "GUBRF.IS", "GWIND.IS", "HALKB.IS", "HEKTS.IS", "IMASM.IS", "IPEKE.IS", "ISCTR.IS", 
        "ISDMR.IS", "ISGYO.IS", "ISMEN.IS", "IZMDC.IS", "KARSN.IS", "KAYSE.IS", "KCAER.IS", "KCHOL.IS", "KONTR.IS", 
        "KONYA.IS", "KOZAA.IS", "KOZAL.IS", "KRDMD.IS", "KZBGY.IS", "MAVI.IS", "MGROS.IS", "MIATK.IS", "ODAS.IS", 
        "OTKAR.IS", "OYAKC.IS", "PENTA.IS", "PETKM.IS", "PGSUS.IS", "PSGYO.IS", "QUAGR.IS", "SAHOL.IS", "SASA.IS", 
        "SELEC.IS", "SISE.IS", "SKBNK.IS", "SMRTG.IS", "SNGYO.IS", "SOKM.IS", "TAVHL.IS", "TCELL.IS", "THYAO.IS", 
        "TKFEN.IS", "TOASO.IS", "TSKB.IS", "TTKOM.IS", "TTRAK.IS", "TUKAS.IS", "TUPRS.IS", "TURSG.IS", "ULKER.IS", 
        "VAKBN.IS", "VESBE.IS", "VESTL.IS", "YEOTK.IS", "YKBNK.IS", "YYLGD.IS", "ZOREN.IS"
    ]

# --- 2. AŞAMA: PROMETHEUS ELEMESİ (Poliklinik) ---
def filter_candidates(tickers):
    print(">>> Aşama 1: Veriler İndiriliyor ve Ön Eleme Yapılıyor...")
    # Son 1 Yıllık Veri
    data = yf.download(tickers, period="1y", group_by='ticker', progress=False)
    
    candidates = []
    
    for ticker in tickers:
        try:
            df = data[ticker]
            if df.empty or len(df) < 200: continue
            
            # --- FİLTRE 1: HACİM (Likitide) ---
            # Son 20 günün ortalama hacmi (TL bazında yaklaşık kontrol)
            avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
            if avg_vol < 10000: continue # Ölü tahtaları at

            # --- FİLTRE 2: TREND (SMA 200) ---
            # 200 Günlük ortalamanın altındaysa "Ayı Piyasası"dır, elenir.
            price = df['Close'].iloc[-1]
            sma200 = df['Close'].rolling(200).mean().iloc[-1]
            
            if price < sma200: continue 

            # --- FİLTRE 3: MOMENTUM (RSI) ---
            # Aşırı şişmişleri (RSI > 80) simülasyona sokma, zaten düşecek.
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            if rsi > 80: continue

            # Elemeyi geçenleri listeye al
            candidates.append({
                "Ticker": ticker,
                "Data": df['Close'] # Kapanış verisini sakla
            })
            
        except: continue
    
    print(f">>> Ön elemeden geçen aday sayısı: {len(candidates)}")
    return candidates

# --- 3. AŞAMA: THE ORACLE (Monte Carlo Simülasyonu) ---
def run_monte_carlo(candidates):
    print(">>> Aşama 2: 10.000 Senaryolu Simülasyon Başlıyor...")
    results = []
    SIMULATIONS = 10000 
    
    for item in candidates:
        ticker = item['Ticker']
        prices = item['Data']
        
        # Getiri İstatistikleri
        returns = prices.pct_change().dropna()
        mu = returns.mean()
        sigma = returns.std()
        start_price = prices.iloc[-1]
        
        # Simülasyon (Vektörel - Hızlı)
        # Gelecek 252 iş günü (1 Yıl)
        sim_returns = np.random.normal(mu, sigma, (252, SIMULATIONS))
        sim_prices = start_price * (1 + sim_returns).cumprod(axis=0)
        final_prices = sim_prices[-1]
        
        # Sonuç Analizi
        loss_count = np.sum(final_prices < start_price)
        prob_loss = (loss_count / SIMULATIONS) * 100
        upside = (np.mean(final_prices) - start_price) / start_price * 100
        
        # Sadece "Gözü Kapalı" güvenli olanları seç (Risk < %35)
        if prob_loss < 35:
            results.append({
                "Hisse": ticker.replace(".IS", ""),
                "Fiyat": start_price,
                "Risk": prob_loss,
                "Potansiyel": upside
            })
            
    return pd.DataFrame(results)

# --- ANA PROGRAM AKIŞI ---
def main():
    send_telegram_message("🚀 Gökhan Hocam, Haftalık 'Çift Aşamalı' Tarama Başladı (Prometheus + Oracle)...")
    
    # 1. Aşama: Listeyi Al ve ELe
    all_tickers = get_bist100_tickers()
    survivors = filter_candidates(all_tickers)
    
    if not survivors:
        send_telegram_message("⚠️ Piyasa 'Ayı Trendi'nde. Hiçbir hisse SMA200 üzerinde değil. Nakitte kal.")
        return

    # 2. Aşama: Simülasyon
    df = run_monte_carlo(survivors)
    
    if not df.empty:
        df = df.sort_values(by='Risk', ascending=True) # En güvenli en üstte
        
        msg = "🦅 *HAFTALIK 'GÖZÜ KAPALI' PORTFÖYÜ*\n"
        msg += f"📅 {datetime.now().strftime('%d-%m-%Y')}\n"
        msg += "Analiz: SMA200 Trend Filtresi + 10.000 Monte Carlo Senaryosu\n\n"
        
        count = 0
        for index, row in df.iterrows():
            if count >= 8: break # En iyi 8
            
            icon = "💎" if row['Risk'] < 10 else "🛡️"
            if row['Potansiyel'] > 80: icon = "🚀"
            
            msg += f"{icon} *{row['Hisse']}* ({row['Fiyat']:.2f} TL)\n"
            msg += f"   Risk: %{row['Risk']:.1f} | Hedef: %{row['Potansiyel']:.0f}\n"
            count += 1
            
        msg += "\n⚠️ _Yapay zeka analizidir._"
        send_telegram_message(msg)
        print("Rapor Telegram'a gönderildi.")
    else:
        send_telegram_message("⚠️ Ön elemeyi geçenler oldu ama Simülasyonda hepsi riskli çıktı. İşlem yapma.")

if __name__ == "__main__":
    main()
