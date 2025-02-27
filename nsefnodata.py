import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from zipfile import ZipFile
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Function to find the previous working day
def previous_working_day(date):
    while date.weekday() >= 5:  # Skip weekends
        date -= timedelta(days=1)
    return date

# Generate working days for past 'months' months
def generate_working_days(start_date, months=6):
    working_days = set()
    for i in range(months * 30):  # Approx 6 months
        day = previous_working_day(start_date - timedelta(days=i))
        if day not in working_days:
            working_days.add(day)
    return sorted(working_days, reverse=True)

# Get today's date and generate working days
today = datetime.now()
working_days = generate_working_days(today, months=6)

# Headers for requests
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Function to download and process BhavCopy in memory
def process_bhavcopy(d):
    date_str = d.strftime('%Y%m%d')
    zip_file = f"BhavCopy_NSE_FO_0_0_0_{date_str}_F_0000.csv.zip"
    url = f"https://nsearchives.nseindia.com/content/fo/{zip_file}"
    print(f"📥 Downloading: {zip_file} from {url}")

    try:
        response = requests.get(url, headers=headers, timeout=15)
        if 'application/zip' in response.headers.get('Content-Type', '').lower():
            with ZipFile(BytesIO(response.content)) as zip_ref:
                csv_name = zip_ref.namelist()[0]
                with zip_ref.open(csv_name) as file:
                    df = pd.read_csv(file, dtype={'SYMBOL': str, 'EXPIRY_DT': str, 'OpnIntrst': float, 'ChngInOpnIntrst': float})
                    df['DATE'] = d.strftime('%Y-%m-%d')  # Add date column
                    return df
        else:
            print(f"⚠️ Skipping {zip_file}, invalid content type.")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading {zip_file}: {e}")
    except Exception as e:
        print(f"❌ Unexpected error processing {zip_file}: {e}")
    return None

# Multithreading for faster downloads
all_data = []
print(f"🚀 Starting multithreaded download for {len(working_days)} BhavCopy files...")
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(process_bhavcopy, d): d for d in working_days}
    for future in as_completed(futures):
        result = future.result()
        if result is not None:
            all_data.append(result)

# Save combined data as CSV
def save_combined_data(dataframes, output_file):
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        if 'OptnTp' in combined_df.columns:
            combined_df = combined_df[~combined_df['OptnTp'].isin(['CE', 'PE'])]
            print(f"🛑 Removed CE & PE rows. Remaining rows: {combined_df.shape[0]}")
        
        # Calculate cumulative Open Interest and Change in Open Interest for each symbol per date
        # Calculate cumulative Open Interest and Change in Open Interest for each symbol per date
        combined_df['Cumulative_OpnIntrst'] = combined_df.groupby(['TradDt', 'TckrSymb'])['OpnIntrst'].transform('sum')
        combined_df['Cumulative_ChngInOpnIntrst'] = combined_df.groupby(['TradDt', 'TckrSymb'])['ChngInOpnIntrst'].transform('sum')
        combined_df['Cumulative_Vol'] = combined_df.groupby(['TradDt', 'TckrSymb'])['TtlTradgVol'].transform('sum')

        # 1️⃣ Calculate FU_LTP_Change and FU_LTP_Change%
        combined_df['FU_LTP_Change'] = combined_df['ClsPric'] - combined_df['PrvsClsgPric']
        combined_df['FU_LTP_Change%'] = (combined_df['FU_LTP_Change'] / combined_df['ClsPric']) * 100

        # 2️⃣ Compute COI%
        combined_df['COI%'] = (combined_df['ChngInOpnIntrst'] / combined_df['OpnIntrst']) * 100
        
         # 2️⃣.1 Compute Cumulative_COI%
        combined_df['Cumulative_COI%'] = (combined_df['Cumulative_ChngInOpnIntrst'] / combined_df['Cumulative_OpnIntrst']) * 100

        # 3️⃣ Compute COIExtension
        combined_df['COIExtension'] = np.abs(combined_df['COI%']) * combined_df['ChngInOpnIntrst']
        
        # 3️⃣.1 Compute COIExtension
        combined_df['Cummu_COIExtension'] = np.abs(combined_df['Cumulative_COI%']) * combined_df['Cumulative_ChngInOpnIntrst']

        # 4️⃣ Compute BuildupStatus (LB, SB, LU, SC)
        conditions = [
            (combined_df['ChngInOpnIntrst'] > 0) & (combined_df['FU_LTP_Change'] > 0),
            (combined_df['ChngInOpnIntrst'] > 0) & (combined_df['FU_LTP_Change'] < 0),
            (combined_df['ChngInOpnIntrst'] < 0) & (combined_df['FU_LTP_Change'] < 0),
            (combined_df['ChngInOpnIntrst'] < 0) & (combined_df['FU_LTP_Change'] > 0)
        ]
        choices = ['LB', 'SB', 'LU', 'SC']
        combined_df['BuildupStatus(Fu)'] = np.select(conditions, choices, default=np.nan)
        
        # 4️⃣.1 Compute BuildupStatus (LB, SB, LU, SC)
        conditions = [
            (combined_df['Cumulative_ChngInOpnIntrst'] > 0) & (combined_df['FU_LTP_Change'] > 0),
            (combined_df['Cumulative_ChngInOpnIntrst'] > 0) & (combined_df['FU_LTP_Change'] < 0),
            (combined_df['Cumulative_ChngInOpnIntrst'] < 0) & (combined_df['FU_LTP_Change'] < 0),
            (combined_df['Cumulative_ChngInOpnIntrst'] < 0) & (combined_df['FU_LTP_Change'] > 0)
        ]
        choices = ['LB', 'SB', 'LU', 'SC']
        combined_df['BuildupStatus(FUc)'] = np.select(conditions, choices, default=np.nan)
        
        combined_df = combined_df.sort_values(by=['TckrSymb', 'XpryDt', 'TradDt'])

        # 5️⃣ Compute 21DSMA_OIT 
        combined_df['21DSMA_OIT(Exp)'] = combined_df.groupby(['TckrSymb', 'XpryDt'])['OpnIntrst'].transform(lambda x: x.rolling(21, min_periods=1).mean())
        
        # 5️⃣ Compute 21DSMA_OIT 
        combined_df['21DSMA_OIc(Exp)'] = combined_df.groupby(['TckrSymb', 'XpryDt'])['Cumulative_OpnIntrst'].transform(lambda x: x.rolling(21, min_periods=1).mean())

        # Ensure 'TradDt' is in datetime format
        combined_df['TradDt'] = pd.to_datetime(combined_df['TradDt'])

        # Sort before applying rolling calculations
        combined_df = combined_df.sort_values(by=['TckrSymb', 'XpryDt', 'TradDt'])

        # Function to compute COIERank using last 21 rows
        def coierank(df):
            df = df.sort_values('TradDt')  # Ensure sorting before rolling calculation
            df['COIExtRank'] = df['COIExtension'].rolling(21, min_periods=1).apply(
                lambda x: ((x[-1] - np.min(x)) / (np.max(x) - np.min(x))) * 100 if np.max(x) != np.min(x) else np.nan,
                raw=True  # NumPy array mode for better performance
            )
            return df

        # Apply the function group-wise
        combined_df = combined_df.groupby(['TckrSymb', 'XpryDt'], group_keys=False).apply(coierank)


         # 6️⃣ Compute COIERank (normalized COIExtension over 21 days)
        def cumm_coierank(df):
            df = df.sort_values('TradDt')
            df['COIExtRankcum'] = df['Cummu_COIExtension'].rolling(21, min_periods=1).apply(
                lambda x: ((x[-1] - x.min()) / (x.max() - x.min())) * 100 if x.max() != x.min() else np.nan,
                raw=True
            )
            return df
        combined_df = combined_df.groupby(['TckrSymb', 'XpryDt'], group_keys=False).apply(cumm_coierank)

         # 7 Compute COIERank (normalized COIExtension over 21 days)
        def coieRankofRank(df):
            df = df.sort_values('TradDt')
            df['COIERankofRank'] = df['COIExtRank'].rolling(21, min_periods=1).apply(
                lambda x: ((x[-1] - x.min()) / (x.max() - x.min())) * 100 if x.max() != x.min() else np.nan,
                raw=True
            )
            return df
        combined_df = combined_df.groupby(['TckrSymb', 'XpryDt'], group_keys=False).apply(coieRankofRank)
        
         # 7.1 Compute COIERank (normalized COIExtension over 21 days)
        def cumm_coieRankofRank(df):
            df = df.sort_values('TradDt')
            df['COIERankofRankcum'] = df['COIExtRankcum'].rolling(21, min_periods=1).apply(
                lambda x: ((x[-1] - x.min()) / (x.max() - x.min())) * 100 if x.max() != x.min() else np.nan,
                raw=True
            )
            return df
        combined_df = combined_df.groupby(['TckrSymb', 'XpryDt'], group_keys=False).apply(cumm_coieRankofRank)
        
         # 8 Compute COIVol
        combined_df['COIpVol'] = (combined_df['COI%'] * combined_df['TtlTradgVol']) 
        combined_df['COIVol'] = (combined_df['ChngInOpnIntrst'] * combined_df['TtlTradgVol'])

         # 8.1 Compute COIVol
        combined_df['Cumm_COIpVol'] = (combined_df['Cumulative_COI%'] * combined_df['Cumulative_Vol']) 
        combined_df['Cumm_COIVol'] = (combined_df['Cumulative_ChngInOpnIntrst'] * combined_df['Cumulative_Vol']) 

         # 9 Compute COIVolRank (normalized  over 21 days)
        def coipcVolrank(df):
            df = df.sort_values('TradDt')
            df['COIpVolRank'] = df['COIpVol'].rolling(21, min_periods=1).apply(
                lambda x: ((x[-1] - x.min()) / (x.max() - x.min())) * 100 if x.max() != x.min() else np.nan,
                raw=True
            )
            return df
        combined_df = combined_df.groupby(['TckrSymb', 'XpryDt'], group_keys=False).apply(coipcVolrank)
        
         # 9.1 Cummu COIVolRank (normalized  over 21 days)
        def cumm_coipcVolrank(df):
            df = df.sort_values('TradDt')
            df['COIpVolRankcum'] = df['Cumm_COIpVol'].rolling(21, min_periods=1).apply(
                lambda x: ((x[-1] - x.min()) / (x.max() - x.min())) * 100 if x.max() != x.min() else np.nan,
                raw=True
            )
            return df
        combined_df = combined_df.groupby(['TckrSymb', 'XpryDt'], group_keys=False).apply(cumm_coipcVolrank)
        

         # 10 Compute COIVolRank (normalized  over 21 days)
        def coicVolrank(df):
            df = df.sort_values('TradDt')
            df['COIVolRank'] = df['COIVol'].rolling(21, min_periods=1).apply(
                lambda x: ((x[-1] - x.min()) / (x.max() - x.min())) * 100 if x.max() != x.min() else np.nan,
                raw=True
            )
            return df
        combined_df = combined_df.groupby(['TckrSymb', 'XpryDt'], group_keys=False).apply(coicVolrank)
        
         # 10.1 Cummu COIVolRank (normalized  over 21 days)
        def cumm_coiVolrank(df):
            df = df.sort_values('TradDt')
            df['COIVolRankcum'] = df['Cumm_COIVol'].rolling(21, min_periods=1).apply(
                lambda x: ((x[-1] - x.min()) / (x.max() - x.min())) * 100 if x.max() != x.min() else np.nan,
                raw=True
            )
            return df
        combined_df = combined_df.groupby(['TckrSymb', 'XpryDt'], group_keys=False).apply(cumm_coiVolrank)

        # 11 Compute COI%
        combined_df['MRank'] = (combined_df['COIERankofRank'] + combined_df['COIVolRank'] + combined_df['COIpVolRank'])/3 
        
        # 11.1 Compute COI%
        combined_df['CummMRank'] = (combined_df['COIERankofRankcum'] + combined_df['COIVolRankcum'] + combined_df['COIpVolRankcum'])/3 
        

        # Convert 'XpryDt' to datetime
        combined_df['XpryDt'] = pd.to_datetime(combined_df['XpryDt'], errors='coerce')

        # Compute Expiry Category
        current_date = pd.Timestamp.today()
        current_month_start = current_date.replace(day=1)
        next_month_start = (current_month_start + pd.DateOffset(months=1)).replace(day=1)
        next_to_next_month_start = (current_month_start + pd.DateOffset(months=2)).replace(day=1)

        combined_df['Expiry Category'] = np.select(
            [
                (combined_df['XpryDt'] >= current_month_start) & (combined_df['XpryDt'] < next_month_start),
                (combined_df['XpryDt'] >= next_month_start) & (combined_df['XpryDt'] < next_to_next_month_start)
            ],
            ['Current Month', 'Next Month'],
            default='Far'
        )

        
        combined_df.to_csv(output_file, index=False)
        print(f"🎉 Data saved as CSV: {output_file}")

# Define output file path
output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "combined_fno_data.csv")
save_combined_data(all_data, output_file)
