
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from matplotlib.ticker import FuncFormatter

sns.set(style='dark')

st.title("Analisis Data Penjualan E-Commerce")
st.markdown("")

@st.cache_data
def load_data(url):
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"Gagal memuat data dari {url}: {e}")
        return pd.DataFrame()

# Dataset URLs
all_data_url = "https://raw.githubusercontent.com/Ardhi-17/Belajar-Analisis-Data-dengan-Python/main/Dataset/all_data.csv"
orders_products_url = "https://raw.githubusercontent.com/Ardhi-17/Belajar-Analisis-Data-dengan-Python/main/Dataset/orders_products_data.csv"

# Load datasets
all_df = load_data(all_data_url)
orders_products_df = load_data(orders_products_url)

def display_clustering_analysis(df):

    st.subheader("hubungan antara harga produk, biaya pengiriman dan skor ulasan pelanggan")

    # Create subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 14))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Scatter plot: Price vs Review Score
    sns.scatterplot(data=df, x='price', y='review_score', alpha=0.6, ax=axes[0, 0],
                    palette='viridis', hue='review_score', legend=False)
    axes[0, 0].set_title('Product Price vs Review Score', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Product Price (BRL)')
    axes[0, 0].set_ylabel('Review Score')
    axes[0, 0].grid(True, alpha=0.3)

    # Scatter plot: Freight Value vs Review Score
    sns.scatterplot(data=df, x='freight_value', y='review_score', alpha=0.6, ax=axes[0, 1],
                    palette='viridis', hue='review_score', legend=False)
    axes[0, 1].set_title('Shipping Cost vs Review Score', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Shipping Cost (BRL)')
    axes[0, 1].set_ylabel('Review Score')
    axes[0, 1].grid(True, alpha=0.3)

    # Prepare price ranges
    bins = [0, 50, 100, 200, 500, 1000, float('inf')]
    labels = ['0-50', '51-100', '101-200', '201-500', '501-1000', '1000+']
    df['price_range'] = pd.cut(df['price'], bins=bins, labels=labels)

    # Bar plot: Average Review Score by Price Range
    price_review = df.groupby('price_range')['review_score'].mean().reset_index()
    price_review.columns = ['price_range', 'mean']

    sns.barplot(data=price_review, x='price_range', y='mean', ax=axes[1, 0], palette='viridis')
    axes[1, 0].set_title('Average Review Score by Price Range', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Price Range (BRL)')
    axes[1, 0].set_ylabel('Average Review Score')
    axes[1, 0].set_ylim(1, 5)
    axes[1, 0].grid(True, alpha=0.3)

    # Prepare freight ranges
    bins = [0, 20, 50, 100, 200, float('inf')]
    labels = ['0-20', '21-50', '51-100', '101-200', '200+']
    df['freight_range'] = pd.cut(df['freight_value'], bins=bins, labels=labels)

    # Bar plot: Average Review Score by Shipping Cost Range
    freight_review = df.groupby('freight_range')['review_score'].mean().reset_index()
    freight_review.columns = ['freight_range', 'mean']

    sns.barplot(data=freight_review, x='freight_range', y='mean', ax=axes[1, 1], palette='viridis')
    axes[1, 1].set_title('Average Review Score by Shipping Cost Range', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Shipping Cost Range (BRL)')
    axes[1, 1].set_ylabel('Average Review Score')
    axes[1, 1].set_ylim(1, 5)
    axes[1, 1].grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()
    st.pyplot(fig)

# Check if data loaded successfully
if all_df.empty or orders_products_df.empty:
    st.stop()

# Sidebar with date range filter and analysis selection
with st.sidebar:
    st.header("Filter dan Pencarian")

    # Date range filter
    st.subheader("Pilih Rentang Waktu")
    date_filter_option = st.radio(
        "Pilih opsi rentang waktu:",
        ("Pilih rentang waktu", "Analisis keseluruhan waktu")
    )

    min_date = pd.to_datetime(all_df['order_purchase_timestamp']).min()
    max_date = pd.to_datetime(all_df['order_purchase_timestamp']).max()

    if date_filter_option == "Pilih rentang waktu":
        start_date, end_date = st.date_input(
            "Pilih rentang waktu:",
            value=[min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
    else:
        start_date = min_date
        end_date = max_date

    # Convert to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Apply filter
    mask = (pd.to_datetime(all_df['order_purchase_timestamp']) >= start_date) & \
           (pd.to_datetime(all_df['order_purchase_timestamp']) <= end_date)
    filtered_all_df = all_df.loc[mask]

    # Check if filtered data is empty
    if filtered_all_df.empty:
        st.warning("Tidak ada data untuk rentang waktu yang dipilih. Silakan pilih rentang waktu yang berbeda.")
        st.stop()

    # Analysis selection
    st.subheader("Pilih Analisis")
    analysis_options = [
        "Tren Penjualan Bulanan",
        "Distribusi Metode Pembayaran",
        "Analisis RFM",
        "Demografi Pelanggan",
        "Pelanggan Terbaik",
        "Metode Pembayaran Paling Digunakan",
        "Analisis Klastering"
    ]
    selected_analysis = st.multiselect("Pilih analisis yang ingin ditampilkan:", analysis_options, default=analysis_options)
# Check if any analysis is selected
if not selected_analysis:
    st.warning("Silakan pilih analisis yang ingin ditampilkan di sidebar.")
else:
    # Main content with tabs
    tabs = st.tabs(selected_analysis)

    # Monthly Sales Trend
    if "Tren Penjualan Bulanan" in selected_analysis:
        with tabs[selected_analysis.index("Tren Penjualan Bulanan")]:
            st.subheader("Tren Penjualan Bulanan")

            if 'order_purchase_timestamp' not in filtered_all_df.columns:
                st.error("Kolom 'order_purchase_timestamp' tidak ditemukan dalam dataset.")
            else:
                try:
                    # Convert to datetime and create 'date' column
                    filtered_all_df['date'] = pd.to_datetime(filtered_all_df['order_purchase_timestamp']).dt.to_period(
                        'M').dt.to_timestamp()

                    # Aggregate monthly data
                    monthly_sales = filtered_all_df.groupby(pd.Grouper(key='date', freq='M')).agg({'price': 'sum'}).reset_index()
                    monthly_sales.rename(columns={'price': 'total_sales'}, inplace=True)

                    # Visualization
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.lineplot(x='date', y='total_sales', data=monthly_sales, marker='o', ax=ax)
                    ax.set_title('Tren Penjualan Bulanan')
                    ax.set_xlabel('Bulan')
                    ax.set_ylabel('Total Penjualan')
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${int(x):,}'))
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Terjadi kesalahan dalam analisis tren penjualan bulanan: {e}")

    # Payment Distribution
    if "Distribusi Metode Pembayaran" in selected_analysis:
        with tabs[selected_analysis.index("Distribusi Metode Pembayaran")]:
            st.subheader("Distribusi Metode Pembayaran")

            if 'payment_type' not in orders_products_df.columns:
                st.error("Kolom 'payment_type' tidak ditemukan dalam dataset orders_products_data.")
            else:
                try:
                    payment_distribution = orders_products_df['payment_type'].value_counts().reset_index()
                    payment_distribution.columns = ['payment_type', 'count']

                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.barplot(x='payment_type', y='count', data=payment_distribution, ax=ax)
                    ax.set_title('Distribusi Metode Pembayaran')
                    ax.set_xlabel('Metode Pembayaran')
                    ax.set_ylabel('Jumlah Transaksi')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Terjadi kesalahan dalam analisis distribusi metode pembayaran: {e}")

    # RFM Analysis
    if "Analisis RFM" in selected_analysis:
        with tabs[selected_analysis.index("Analisis RFM")]:
            st.subheader("Analisis RFM (Recency, Frequency, Monetary)")

            required_columns = ['customer_id', 'order_id', 'payment_value', 'order_purchase_timestamp']
            missing_cols = [col for col in required_columns if col not in filtered_all_df.columns]

            if missing_cols:
                st.error(f"Dataset tidak memiliki kolom yang diperlukan untuk analisis RFM: {', '.join(missing_cols)}")
            else:
                try:
                    # Remove rows with missing values
                    filtered_all_df_rfm = filtered_all_df.dropna(subset=required_columns)

                    # Ensure 'order_purchase_timestamp' is datetime type
                    filtered_all_df_rfm['order_purchase_timestamp'] = pd.to_datetime(
                        filtered_all_df_rfm['order_purchase_timestamp'],
                        errors='coerce')

                    # Check if datetime conversion was successful
                    if filtered_all_df_rfm['order_purchase_timestamp'].isna().any():
                        st.error(
                            "Ada nilai yang tidak valid dalam kolom 'order_purchase_timestamp'. Periksa format tanggal.")
                    else:
                        # Calculate RFM metrics
                        now = filtered_all_df_rfm['order_purchase_timestamp'].max()

                        rfm_df = filtered_all_df_rfm.groupby('customer_id').agg({
                            'order_purchase_timestamp': lambda x: (now - x.max()).days,
                            'order_id': 'nunique',
                            'payment_value': 'sum'
                        }).reset_index()

                        rfm_df.columns = ['customer_id', 'recency', 'frequency', 'monetary']

                        # Create numeric ID for customers
                        rfm_df['numeric_id'] = pd.factorize(rfm_df['customer_id'])[0] + 1

                        # Check if there's data to display
                        if rfm_df.empty:
                            st.warning("Tidak ada data untuk analisis RFM dengan filter saat ini.")
                        else:
                            # Create figure with subplots
                            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 6))
                            colors = sns.color_palette("husl", 5)

                            # Bar plot for Recency (Last Purchase)
                            sorted_recency = rfm_df.sort_values(by='recency', ascending=True).head(5)
                            sns.barplot(
                                y="recency",
                                x='numeric_id',
                                data=sorted_recency,
                                hue='numeric_id',
                                palette=colors,
                                ax=ax[0],
                                dodge=False,
                                width=0.5
                            )
                            ax[0].set_ylabel('Days since last purchase')
                            ax[0].set_xlabel('Customer ID')
                            ax[0].set_title('Last Purchase (days)', loc='center', fontsize=18, fontweight='bold')
                            ax[0].tick_params(axis='x', labelsize=15)
                            ax[0].tick_params(axis='y', labelsize=12)
                            ax[0].legend(title='Customer ID', fontsize=12, title_fontsize=14, loc='upper right',
                                         bbox_to_anchor=(1.15, 1))

                            # Bar plot for Frequency (Number of Purchases)
                            sorted_frequency = rfm_df.sort_values(by='frequency', ascending=False).head(5)
                            sns.barplot(
                                y='frequency',
                                x='numeric_id',
                                data=sorted_frequency,
                                hue='numeric_id',
                                palette=colors,
                                ax=ax[1],
                                dodge=False,
                                width=0.5
                            )
                            ax[1].set_ylabel('Number of purchases')
                            ax[1].set_xlabel('Customer ID')
                            ax[1].set_title('Purchase Frequency', loc='center', fontsize=18, fontweight='bold')
                            ax[1].tick_params(axis='x', labelsize=15)
                            ax[1].tick_params(axis='y', labelsize=12)
                            ax[1].legend(title='Customer ID', fontsize=12, title_fontsize=14, loc='upper right',
                                         bbox_to_anchor=(1.15, 1))

                            # Bar plot for Monetary (Total Money Spent)
                            sorted_monetary = rfm_df.sort_values(by='monetary', ascending=False).head(5)
                            sns.barplot(
                                y='monetary',
                                x='numeric_id',
                                data=sorted_monetary,
                                hue='numeric_id',
                                palette=colors,
                                ax=ax[2],
                                dodge=False,
                                width=0.5
                            )
                            ax[2].set_ylabel('Total amount spent')
                            ax[2].set_xlabel('Customer ID')
                            ax[2].set_title('Total Amount Spent', loc='center', fontsize=18, fontweight='bold')
                            ax[2].tick_params(axis='x', labelsize=15)
                            ax[2].tick_params(axis='y', labelsize=12)
                            ax[2].legend(title='Customer ID', fontsize=12, title_fontsize=14, loc='upper right',
                                         bbox_to_anchor=(1.15, 1))

                            plt.suptitle('Best Customers Based on RFM Parameters', fontsize=22, fontweight='bold')
                            plt.tight_layout()
                            st.pyplot(fig)

                            # Display RFM data
                            st.write("### Data RFM (5 Pelanggan Terbaik)")
                            st.dataframe(
                                rfm_df.sort_values(by=['recency', 'frequency', 'monetary'],
                                                   ascending=[True, False, False]).head(5))
                except Exception as e:
                    st.error(f"Terjadi kesalahan dalam analisis RFM: {e}")

    # Customer Demographic
    if "Demografi Pelanggan" in selected_analysis:
        with tabs[selected_analysis.index("Demografi Pelanggan")]:
            st.subheader("Demografi Pelanggan")

            # Tabs for state analysis and geolocation
            demographic_tabs = st.tabs(["Per Provinsi", "Peta Sebaran"])

            with demographic_tabs[0]:
                if 'customer_state' not in filtered_all_df.columns:
                    st.error("Kolom 'customer_state' tidak ditemukan dalam dataset.")
                else:
                    try:
                        # Calculate customer distribution per state
                        state_distribution = filtered_all_df['customer_state'].value_counts().reset_index()
                        state_distribution.columns = ['customer_state', 'customer_count']

                        # Display state with most customers
                        most_common_state = state_distribution.iloc[0]['customer_state']
                        st.markdown(f"**Provinsi dengan Pelanggan Terbanyak:** {most_common_state}")

                        # Visualization
                        fig, ax = plt.subplots(figsize=(12, 6))
                        sns.barplot(
                            x=state_distribution['customer_state'],
                            y=state_distribution['customer_count'],
                            palette="viridis"
                        )
                        ax.set_title("Jumlah Pelanggan per Provinsi")
                        ax.set_xlabel("Provinsi")
                        ax.set_ylabel("Jumlah Pelanggan")
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Terjadi kesalahan dalam analisis demografi pelanggan: {e}")

            with demographic_tabs[1]:
                if 'geolocation_lat' not in filtered_all_df.columns or 'geolocation_lng' not in filtered_all_df.columns:
                    st.warning("Kolom koordinat geografis tidak ditemukan. Fitur peta tidak tersedia.")
                else:
                    try:
                        # Create customer distribution map
                        st.map(filtered_all_df[['geolocation_lat', 'geolocation_lng']].dropna())

                        # Add explanation
                        with st.expander("Lihat Penjelasan"):
                            st.write("""
                            Berdasarkan peta di atas, dapat dilihat bahwa sebagian besar pelanggan terkonsentrasi di wilayah-wilayah:
                            - Bagian selatan dan tenggara
                            - Kota-kota besar dan ibu kota provinsi

                            Hal ini menunjukkan bahwa strategi pemasaran yang lebih efektif dapat difokuskan pada wilayah-wilayah dengan densitas pelanggan tinggi.
                            """)
                    except Exception as e:
                        st.error(f"Terjadi kesalahan dalam analisis peta sebaran pelanggan: {e}")

    # Best Customers
    if "Pelanggan Terbaik" in selected_analysis:
        with tabs[selected_analysis.index("Pelanggan Terbaik")]:
            st.subheader("Pelanggan Terbaik")

            if 'customer_id' not in filtered_all_df.columns or 'order_id' not in filtered_all_df.columns or 'payment_value' not in filtered_all_df.columns:
                st.error("Kolom yang diperlukan untuk analisis pelanggan terbaik tidak ditemukan.")
            else:
                try:
                    # Calculate best customers
                    best_customers = filtered_all_df.groupby('customer_id').agg({
                        'order_id': 'nunique',
                        'payment_value': 'sum'
                    }).reset_index()
                    best_customers.columns = ['customer_id', 'total_orders', 'total_spent']

                    # Sort by total spent
                    best_customers = best_customers.sort_values(by='total_spent', ascending=False).head(10)

                    # Visualization
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.barplot(x='customer_id', y='total_spent', data=best_customers, palette='viridis')
                    ax.set_title('Top 10 Pelanggan dengan Pengeluaran Tertinggi')
                    ax.set_xlabel('Customer ID')
                    ax.set_ylabel('Total Pengeluaran')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                    # Display data
                    st.write("### Data Pelanggan Terbaik")
                    st.dataframe(best_customers)
                except Exception as e:
                    st.error(f"Terjadi kesalahan dalam analisis pelanggan terbaik: {e}")

    # Most Used Payment Method
    if "Metode Pembayaran Paling Digunakan" in selected_analysis:
        with tabs[selected_analysis.index("Metode Pembayaran Paling Digunakan")]:
            st.subheader("Metode Pembayaran Paling Digunakan")

            if 'payment_type' not in orders_products_df.columns:
                st.error("Kolom 'payment_type' tidak ditemukan dalam dataset orders_products_data.")
            else:
                try:
                    payment_distribution = orders_products_df['payment_type'].value_counts().reset_index()
                    payment_distribution.columns = ['payment_type', 'count']

                    # Visualization
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.barplot(x='payment_type', y='count', data=payment_distribution, ax=ax)
                    ax.set_title('Distribusi Metode Pembayaran')
                    ax.set_xlabel('Metode Pembayaran')
                    ax.set_ylabel('Jumlah Transaksi')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                    # Display data
                    st.write("### Data Metode Pembayaran")
                    st.dataframe(payment_distribution)
                except Exception as e:
                    st.error(f"Terjadi kesalahan dalam analisis metode pembayaran paling digunakan: {e}")
# In your analysis selection
    if "Analisis Klastering" in selected_analysis:
         with tabs[selected_analysis.index("Analisis Klastering")]:
             display_clustering_analysis(filtered_all_df)

st.caption('Copyright © Ahcmad Ardhi Arridho')