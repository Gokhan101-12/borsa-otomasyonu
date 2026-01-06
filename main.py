import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import random
from datetime import datetime

# --- 0. TELEGRAM AYARLARI ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
        requests.post(url, json=payload)
    except: pass

# --- 1. BIST 100 LİSTESİ ---
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

# --- 2. EVRİMSEL MOTOR (Genetik Algoritma) ---
def calculate_indicators(df):
    # Robotun gözlerini (sensörlerini) oluştur
    df = df.copy()
    # Trend Sensörleri
    df['Trend_50'] = df['Close'] / df['Close'].rolling(50).mean()
    df['Trend_200'] = df['Close'] / df['Close'].rolling(200).mean()
    
    # RSI Sensörü
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI_Norm'] = (100 - (100 / (1 + rs))) / 100
    
    # Volatilite Sensörü
    df['Volat'] = (df['High'] - df['Low']) / df['Close']
    
    return df.dropna()

def fitness_function(genes, df):
    # Verilen genlere (ağırlıklara) göre kar/zarar hesapla
    # Sinyal = (w1*T50) + (w2*T200) + (w3*RSI) + (w4*Volat)
    signals = (
        (genes[0] * df['Trend_50']) + 
        (genes[1] * df['Trend_200']) + 
        (genes[2] * df['RSI_Norm']) + 
        (genes[3] * df['Volat'])
    )
    
    # Basit Backtest: Sinyal > Eşik ise AL
    # Hızlı hesaplama için vektörel işlem
    buy_threshold = genes[4]
    
    # Sinyal oluştuğu günlerin ertesi gün getirisi
    returns = df['Close'].pct_change().shift(-1)
    
    # Pozisyonda olduğumuz günlerin getirisi
    strategy_returns = returns[signals > buy_threshold]
    
    if len(strategy_returns) == 0: return -1 # Hiç işlem yapmadıysa kötü
    return strategy_returns.sum() # Toplam getiri skoru

def evolve_species(df):
    # HIZLI EVRİM (GitHub sunucusu yorulmasın diye optimize edildi)
    POP_SIZE = 20   # Popülasyon
    GENS = 10       # Nesil Sayısı
    
    # Rastgele ilk nesil [w1, w2, w3, w4, buy_thresh]
    population = [np.random.uniform(-1, 1, 5) for _ in range(POP_SIZE)]
    
    best_genes = None
    best_score = -999
    
    for _ in range(GENS):
        # Herkesi test et
        scores = [(fitness_function(dna, df), dna) for dna in population]
        scores.sort(key=lambda x: x[0], reverse=True)
        
        if scores[0][0] > best_score:
            best_score = scores[0][0]
            best_genes = scores[0][1]
            
        # Doğal Seçilim (En iyi %50 yaşar)
        survivors = [s[1] for s in scores[:POP_SIZE//2]]
        
        # Üreme ve Mutasyon
        next_gen = survivors[:]
        while len(next_gen) < POP_SIZE:
            parent = random.choice(survivors)
            child = parent + np.random.normal(0, 0.1, 5) # Küçük mutasyon
            next_gen.append(child)
        population = next_gen
        
    return best_genes, best_score

# --- 3. AŞAMA: TARAMA VE CANLI SİNYAL ---
def scan_market(tickers):
    print(">>> Veriler İndiriliyor ve Her Hisse İçin Evrim Başlatılıyor...")
    data = yf.download(tickers, period="1y", group_by='ticker', progress=False)
    
    candidates = []
    
    for ticker in tickers:
        try:
            df = data[ticker]
            if df.empty or len(df) < 200: continue
            
            # Hacim Kontrolü (Ölü hisse eleme)
            if df['Volume'].iloc[-1] * df['Close'].iloc[-1] < 5000000: continue

            # İndikatörleri hazırla
            df_ind = calculate_indicators(df)
            
            # EVRİM: Bu hisse için en iyi DNA'yı bul
            # Son 1 yılın ilk 10 ayını eğitim, son 2 ayını test gibi düşünebiliriz
            # Ama basitlik için tüm veride en iyiyi buluyoruz.
            best_dna, score = evolve_species(df_ind)
            
            # Eğer robot geçmişte para kazanamadıysa (Skor < 0), bu hisse ölüdür.
            if score < 0.10: continue # %10 altı getiri üreten stratejiyi at
            
            # CANLI SİNYAL KONTROLÜ
            # Bulunan "Kazanan DNA" bugün AL veriyor mu?
            last_row = df_ind.iloc[-1]
            signal_strength = (
                (best_dna[0] * last_row['Trend_50']) + 
                (best_dna[1] * last_row['Trend_200']) + 
                (best_dna[2] * last_row['RSI_Norm']) + 
                (best_dna[3] * last_row['Volat'])
            )
            
            if signal_strength > best_dna[4]: # Eşiği geçtiyse
                candidates.append({
                    "Ticker": ticker,
                    "Fiyat": last_row['Close'],
                    "Data": df['Close'],
                    "DNA_Skoru": score
                })
                
        except: continue
        
    print(f">>> Evrimsel elemeyi geçen aday sayısı: {len(candidates)}")
    return candidates

# --- 4. AŞAMA: THE ORACLE (Monte Carlo) ---
def run_monte_carlo(candidates):
    print(">>> Monte Carlo Stres Testi...")
    results = []
    SIMULATIONS = 5000 
    
    for item in candidates:
        ticker = item['Ticker']
        prices = item['Data']
        
        returns = prices.pct_change().dropna()
        mu = returns.mean()
        sigma = returns.std()
        start_price = prices.iloc[-1]
        
        sim_returns = np.random.normal(mu, sigma, (252, SIMULATIONS))
        sim_prices = start_price * (1 + sim_returns).cumprod(axis=0)
        final_prices = sim_prices[-1]
        
        loss_prob = (np.sum(final_prices < start_price) / SIMULATIONS) * 100
        upside = (np.mean(final_prices) - start_price) / start_price * 100
        
        # Sadece Güvenli Olanlar (Risk < %35)
        if loss_prob < 35:
            results.append({
                "Hisse": ticker.replace(".IS", ""),
                "Fiyat": start_price,
                "Strateji_Puani": item['DNA_Skoru'],
                "Risk": loss_prob,
                "Potansiyel": upside
            })
            
    return pd.DataFrame(results)

# --- ANA PROGRAM ---
def main():
    send_telegram_message("🧬 Gökhan Hocam, 'PROJECT DARWIN' (Kişiye Özel Evrim) Taraması Başladı...")
    
    tickers = get_bist100_tickers()
    survivors = scan_market(tickers)
    
    if not survivors:
        send_telegram_message("⚠️ Piyasa çok zorlu. Hiçbir hisse için karlı bir genetik formül bulunamadı.")
        return

    df = run_monte_carlo(survivors)
    
    if not df.empty:
        df = df.sort_values(by='Risk', ascending=True)
        
        msg = "🦁 *HAFTALIK EVRİMSEL PORTFÖY*\n"
        msg += f"📅 {datetime.now().strftime('%d-%m-%Y')}\n"
        msg += "Yöntem: Genetik Algoritma (Her hisseye özel formül) + Monte Carlo\n\n"
        
        count = 0
        for index, row in df.iterrows():
            if count >= 8: break
            
            icon = "💎" if row['Risk'] < 5 else "🛡️"
            if row['Potansiyel'] > 60: icon = "🚀"
            
            msg += f"{icon} *{row['Hisse']}* ({row['Fiyat']:.2f} TL)\n"
            msg += f"   🧬 Genetik Skor: {row['Strateji_Puani']:.2f}\n"
            msg += f"   📉 Risk: %{row['Risk']:.1f} | 📈 Hedef: %{row['Potansiyel']:.0f}\n\n"
            count += 1
            
        msg += "⚠️ _Yapay zeka analizidir._"
        send_telegram_message(msg)
        print("Rapor gönderildi.")
    else:
        send_telegram_message("⚠️ Genetik sinyal var ama Monte Carlo'da riskli çıktılar. İşlem yapma.")

if __name__ == "__main__":
    main()
