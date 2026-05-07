import pandas as pd

def load_and_prepare_data(path):

    df = pd.read_csv(path)

    # Robust mixed-format date parsing
    df['Date'] = pd.to_datetime(
        df['Date'],
        format='mixed',
        dayfirst=True,
        errors='coerce'
    )

    df = df.dropna(subset=['Date'])

    # Clean numeric values
    df['Total'] = (
        df['Total']
        .astype(str)
        .str.replace(',', '', regex=False)
        .astype(float)
    )

    df = df.sort_values(['State', 'Date'])

    processed = []

    for state in df['State'].unique():

        temp = df[df['State'] == state].copy()

        temp = temp[['Date', 'State', 'Total']]

        temp = temp.set_index('Date').asfreq('W')

        temp['State'] = state

        temp['Total'] = (
            temp['Total']
            .interpolate(method='linear')
            .bfill()
            .ffill()
        )

        temp = temp.reset_index()

        temp.columns = ['Date', 'State', 'Total']

        processed.append(temp)

    final_df = pd.concat(processed, ignore_index=True)

    return final_df
