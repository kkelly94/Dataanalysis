import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Data Visualization & Analysis Tool",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = None
if 'normalized_columns' not in st.session_state:
    st.session_state.normalized_columns = {}

def load_data(uploaded_file):
    """Load data from uploaded CSV or Excel file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            data = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel files.")
            return None
        
        # Detect and convert datetime columns
        for col in data.columns:
            if data[col].dtype == 'object':
                try:
                    data[col] = pd.to_datetime(data[col], errors='ignore')
                except:
                    pass
        
        return data
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def get_column_info(data):
    """Get information about column types"""
    info = {}
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            info[col] = 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(data[col]):
            info[col] = 'datetime'
        else:
            info[col] = 'categorical'
    return info

def calculate_moving_average(data, column, window, ma_type='simple', time_col=None):
    """Calculate moving average with time-based window support"""
    if time_col and pd.api.types.is_datetime64_any_dtype(data[time_col]):
        # Use time-based rolling window
        data_indexed = data.set_index(time_col).sort_index()
        if ma_type == 'simple':
            return data_indexed[column].rolling(window).mean()
        elif ma_type == 'exponential':
            return data_indexed[column].ewm(span=window).mean()
        else:
            return data_indexed[column].rolling(window).mean()
    else:
        # Use numeric window
        if ma_type == 'simple':
            return data[column].rolling(window=window).mean()
        elif ma_type == 'exponential':
            return data[column].ewm(span=window).mean()
        else:
            return data[column].rolling(window=window).mean()

def calculate_bollinger_bands(data, column, window=20, num_std=2, time_col=None):
    """Calculate Bollinger Bands with time-based window support"""
    if time_col and pd.api.types.is_datetime64_any_dtype(data[time_col]):
        # Use time-based rolling window
        data_indexed = data.set_index(time_col).sort_index()
        rolling_mean = data_indexed[column].rolling(window).mean()
        rolling_std = data_indexed[column].rolling(window).std()
    else:
        # Use numeric window
        rolling_mean = data[column].rolling(window=window).mean()
        rolling_std = data[column].rolling(window=window).std()
    
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    
    return rolling_mean, upper_band, lower_band

def create_plot(data, x_col, y_col, chart_type, color_col=None):
    """Create interactive plot using Plotly"""
    try:
        if chart_type == 'Scatter Plot':
            fig = px.scatter(data, x=x_col, y=y_col, color=color_col, 
                           title=f'{y_col} vs {x_col}',
                           hover_data=[col for col in data.columns if col not in [x_col, y_col]])
        elif chart_type == 'Line Chart':
            fig = px.line(data, x=x_col, y=y_col, color=color_col,
                         title=f'{y_col} over {x_col}')
        elif chart_type == 'Bar Chart':
            if color_col:
                fig = px.bar(data, x=x_col, y=y_col, color=color_col,
                           title=f'{y_col} by {x_col}')
            else:
                fig = px.bar(data, x=x_col, y=y_col,
                           title=f'{y_col} by {x_col}')
        else:
            fig = px.scatter(data, x=x_col, y=y_col, color=color_col,
                           title=f'{y_col} vs {x_col}')
        
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            hovermode='closest'
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")
        return None

def add_statistical_overlays(fig, data, x_col, y_col, show_ma=False, ma_window=20, ma_type='simple',
                           show_bollinger=False, bollinger_window=20, bollinger_std=2):
    """Add statistical overlays to the plot"""
    if not pd.api.types.is_numeric_dtype(data[y_col]):
        return fig
    
    # Sort data by x-axis for proper overlay alignment
    data_sorted = data.sort_values(x_col).reset_index(drop=True)
    
    # Determine if we should use time-based rolling
    use_time_based = pd.api.types.is_datetime64_any_dtype(data_sorted[x_col])
    
    if show_ma:
        if use_time_based:
            # Use time-based rolling window (e.g., '365D' for 1 year)
            window_str = f"{ma_window}D"
            ma_data = calculate_moving_average(data_sorted, y_col, window_str, ma_type, x_col)
            # Get the index values for plotting
            x_values = ma_data.index
            y_values = ma_data.values
        else:
            # Use numeric window
            ma_data = calculate_moving_average(data_sorted, y_col, ma_window, ma_type)
            x_values = data_sorted[x_col]
            y_values = ma_data
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines',
            name=f'{ma_type.title()} MA ({ma_window}{"D" if use_time_based else ""})',
            line=dict(color='gray', width=2)
        ))
    
    if show_bollinger:
        if use_time_based:
            # Use time-based rolling window
            window_str = f"{bollinger_window}D"
            rolling_mean, upper_band, lower_band = calculate_bollinger_bands(
                data_sorted, y_col, window_str, bollinger_std, x_col)
            # Get the index values for plotting
            x_values = rolling_mean.index
            mean_values = rolling_mean.values
            upper_values = upper_band.values
            lower_values = lower_band.values
        else:
            # Use numeric window
            rolling_mean, upper_band, lower_band = calculate_bollinger_bands(
                data_sorted, y_col, bollinger_window, bollinger_std)
            x_values = data_sorted[x_col]
            mean_values = rolling_mean
            upper_values = upper_band
            lower_values = lower_band
        
        # Calculate standard deviation for proper bands
        if use_time_based:
            data_indexed = data_sorted.set_index(x_col).sort_index()
            rolling_std = data_indexed[y_col].rolling(window_str).std()
            std_values = rolling_std.values
        else:
            rolling_std = data_sorted[y_col].rolling(window=bollinger_window).std()
            std_values = rolling_std
        
        # Add ±2 standard deviation band (outer, lighter)
        fig.add_trace(go.Scatter(
            x=x_values,
            y=mean_values + 2*std_values,
            mode='lines',
            name=f'±2 Std Dev',
            line=dict(color='rgba(100,100,100,0)', width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=mean_values - 2*std_values,
            mode='lines',
            name=f'±2 Std Dev',
            line=dict(color='rgba(100,100,100,0)', width=0),
            fill='tonexty',
            fillcolor='rgba(100,100,100,0.2)',
            showlegend=True
        ))
        
        # Add ±1 standard deviation band (inner, darker)
        fig.add_trace(go.Scatter(
            x=x_values,
            y=mean_values + std_values,
            mode='lines',
            name=f'±1 Std Dev',
            line=dict(color='rgba(100,100,100,0)', width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=mean_values - std_values,
            mode='lines',
            name=f'±1 Std Dev',
            line=dict(color='rgba(100,100,100,0)', width=0),
            fill='tonexty',
            fillcolor='rgba(100,100,100,0.4)',
            showlegend=True
        ))
        
        # Add rolling mean line
        fig.add_trace(go.Scatter(
            x=x_values,
            y=mean_values,
            mode='lines',
            name=f'Rolling Mean ({bollinger_window}{"D" if use_time_based else ""})',
            line=dict(color='gray', width=2, dash='dot')
        ))
    
    return fig

def apply_filters(data, filters):
    """Apply filters to the data"""
    filtered_data = data.copy()
    
    for col, filter_config in filters.items():
        if filter_config['type'] == 'numeric':
            min_val, max_val = filter_config['range']
            filtered_data = filtered_data[
                (filtered_data[col] >= min_val) & (filtered_data[col] <= max_val)
            ]
        elif filter_config['type'] == 'categorical':
            selected_values = filter_config['values']
            if selected_values:
                filtered_data = filtered_data[filtered_data[col].isin(selected_values)]
        elif filter_config['type'] == 'datetime':
            start_date, end_date = filter_config['range']
            # Convert date objects to datetime for comparison
            start_datetime = pd.to_datetime(start_date)
            end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            filtered_data = filtered_data[
                (filtered_data[col] >= start_datetime) & (filtered_data[col] <= end_datetime)
            ]
    
    return filtered_data

def create_normalized_column(data, numerator_col, denominator_col):
    """Create a normalized column by dividing one column by another"""
    try:
        # Avoid division by zero
        normalized_data = data[numerator_col] / data[denominator_col].replace(0, np.nan)
        return normalized_data
    except Exception as e:
        st.error(f"Error creating normalized column: {str(e)}")
        return None

def display_statistics(data, column):
    """Display statistical summary for a column"""
    if pd.api.types.is_numeric_dtype(data[column]):
        stats = data[column].describe()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean", f"{stats['mean']:.2f}")
            st.metric("Median", f"{stats['50%']:.2f}")
            st.metric("Mode", f"{data[column].mode().iloc[0]:.2f}" if not data[column].mode().empty else "N/A")
        
        with col2:
            st.metric("Std Dev", f"{stats['std']:.2f}")
            st.metric("Variance", f"{data[column].var():.2f}")
            st.metric("Skewness", f"{data[column].skew():.2f}")
        
        with col3:
            st.metric("Min", f"{stats['min']:.2f}")
            st.metric("Max", f"{stats['max']:.2f}")
            st.metric("Range", f"{stats['max'] - stats['min']:.2f}")
        
        # Additional statistics
        st.subheader("Additional Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Kurtosis", f"{data[column].kurtosis():.2f}")
            st.metric("25th Percentile", f"{stats['25%']:.2f}")
        with col2:
            st.metric("75th Percentile", f"{stats['75%']:.2f}")
            st.metric("IQR", f"{stats['75%'] - stats['25%']:.2f}")
    else:
        st.write("Statistical analysis not available for non-numeric columns")

# Main application
def main():
    st.title("📊 Data Visualization & Analysis Tool")
    st.markdown("Upload your CSV or Excel data and explore it with interactive visualizations and statistical analysis.")
    
    # Sidebar for controls
    st.sidebar.header("🔧 Controls")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your data file to get started"
    )
    
    if uploaded_file is not None:
        # Load data
        if st.session_state.data is None or st.sidebar.button("🔄 Reload Data"):
            with st.spinner("Loading data..."):
                st.session_state.data = load_data(uploaded_file)
        
        if st.session_state.data is not None:
            data = st.session_state.data
            column_info = get_column_info(data)
            
            # Data overview
            st.header("📋 Data Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", len(data))
            with col2:
                st.metric("Total Columns", len(data.columns))
            with col3:
                st.metric("Missing Values", data.isnull().sum().sum())
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(data.head(100), use_container_width=True)
            
            # Column information
            st.subheader("Column Information")
            col_info_df = pd.DataFrame([
                {
                    'Column': col,
                    'Type': column_info[col],
                    'Non-Null Count': data[col].count(),
                    'Null Count': data[col].isnull().sum(),
                    'Unique Values': data[col].nunique()
                }
                for col in data.columns
            ])
            st.dataframe(col_info_df, use_container_width=True)
            
            # Filters
            st.sidebar.subheader("🔍 Filters")
            filters = {}
            
            for col in data.columns:
                if column_info[col] == 'numeric':
                    min_val, max_val = float(data[col].min()), float(data[col].max())
                    if min_val != max_val:
                        range_val = st.sidebar.slider(
                            f"Filter {col}",
                            min_value=min_val,
                            max_value=max_val,
                            value=(min_val, max_val),
                            key=f"filter_{col}"
                        )
                        filters[col] = {'type': 'numeric', 'range': range_val}
                
                elif column_info[col] == 'categorical':
                    unique_values = data[col].unique()
                    if len(unique_values) <= 20:  # Only show filter for reasonable number of categories
                        st.sidebar.write(f"**Filter {col}:**")
                        selected_values = []
                        
                        # Create checkboxes for each unique value
                        for value in unique_values:
                            if st.sidebar.checkbox(
                                f"{value}",
                                value=True,  # Default checked
                                key=f"filter_{col}_{value}"
                            ):
                                selected_values.append(value)
                        
                        filters[col] = {'type': 'categorical', 'values': selected_values}
                
                elif column_info[col] == 'datetime':
                    min_date, max_date = data[col].min(), data[col].max()
                    if pd.notna(min_date) and pd.notna(max_date):
                        date_range = st.sidebar.date_input(
                            f"Filter {col}",
                            value=(min_date.date(), max_date.date()),
                            min_value=min_date.date(),
                            max_value=max_date.date(),
                            key=f"filter_{col}"
                        )
                        if len(date_range) == 2:
                            filters[col] = {'type': 'datetime', 'range': date_range}
            
            # Apply filters
            if filters:
                filtered_data = apply_filters(data, filters)
                st.session_state.filtered_data = filtered_data
            else:
                st.session_state.filtered_data = data
            
            filtered_data = st.session_state.filtered_data
            
            if len(filtered_data) == 0:
                st.warning("No data matches the current filters. Please adjust your filter settings.")
                return
            
            # Visualization section
            st.header("📈 Data Visualization")
            
            # Plot configuration
            col1, col2 = st.columns(2)
            
            with col1:
                # Axis selection
                numeric_cols = [col for col, info in column_info.items() if info in ['numeric', 'datetime']]
                all_cols = list(data.columns)
                
                x_col = st.selectbox("Select X-axis", options=all_cols, key="x_axis")
                
                # Y-axis selection with normalization option
                st.subheader("Y-axis Configuration")
                use_normalization = st.checkbox("Normalize Y-axis data", key="use_normalization")
                
                if use_normalization:
                    numerator_col = st.selectbox("Numerator column", options=numeric_cols, key="numerator_col")
                    denominator_col = st.selectbox("Denominator column", options=numeric_cols, key="denominator_col")
                    
                    if numerator_col and denominator_col and numerator_col != denominator_col:
                        # Create normalized column
                        normalized_data = create_normalized_column(filtered_data, numerator_col, denominator_col)
                        if normalized_data is not None:
                            filtered_data = filtered_data.copy()
                            normalized_col_name = f"{numerator_col}_per_{denominator_col}"
                            filtered_data[normalized_col_name] = normalized_data
                            y_col = normalized_col_name
                            
                            # Save to session state for later use
                            st.session_state.normalized_columns[normalized_col_name] = {
                                'numerator': numerator_col,
                                'denominator': denominator_col,
                                'data': normalized_data
                            }
                            
                            st.success(f"Created normalized column: {normalized_col_name}")
                        else:
                            y_col = st.selectbox("Select Y-axis", options=numeric_cols, key="y_axis_fallback")
                    else:
                        y_col = st.selectbox("Select Y-axis", options=numeric_cols, key="y_axis_fallback2")
                else:
                    y_col = st.selectbox("Select Y-axis", options=numeric_cols, key="y_axis")
                
                # Optional color column
                color_col = st.selectbox(
                    "Color by (optional)",
                    options=[None] + [col for col in all_cols if data[col].nunique() <= 20],
                    key="color_col"
                )
                
                # Chart type
                chart_type = st.selectbox(
                    "Chart Type",
                    ["Scatter Plot", "Line Chart", "Bar Chart"],
                    key="chart_type"
                )
            
            with col2:
                # Statistical overlays
                st.subheader("Statistical Overlays")
                
                show_ma = st.checkbox("Show Moving Average", key="show_ma")
                if show_ma:
                    # Check if x-axis is datetime for time-based windows
                    if x_col and pd.api.types.is_datetime64_any_dtype(filtered_data[x_col]):
                        ma_window = st.slider("MA Window (Days)", min_value=1, max_value=3650, value=365, key="ma_window")
                        st.caption("Time-based rolling window (e.g., 365 = 1 year)")
                    else:
                        ma_window = st.slider("MA Window (Points)", min_value=2, max_value=100, value=20, key="ma_window")
                        st.caption("Point-based rolling window")
                    ma_type = st.selectbox("MA Type", ["simple", "exponential"], key="ma_type")
                else:
                    ma_window, ma_type = 365, 'simple'
                
                show_bollinger = st.checkbox("Show Standard Deviation Bands", key="show_bollinger")
                if show_bollinger:
                    # Check if x-axis is datetime for time-based windows
                    if x_col and pd.api.types.is_datetime64_any_dtype(filtered_data[x_col]):
                        bollinger_window = st.slider("Rolling Window (Days)", min_value=1, max_value=3650, value=365, key="bollinger_window")
                        st.caption("Time-based rolling window (e.g., 365 = 1 year)")
                    else:
                        bollinger_window = st.slider("Rolling Window (Points)", min_value=2, max_value=100, value=20, key="bollinger_window")
                        st.caption("Point-based rolling window")
                    bollinger_std = st.slider("Standard Deviations", min_value=1.0, max_value=3.0, value=1.0, step=0.5, key="bollinger_std")
                else:
                    bollinger_window, bollinger_std = 365, 1.0
            
            # Create and display plot
            if x_col and y_col:
                fig = create_plot(filtered_data, x_col, y_col, chart_type, color_col)
                
                if fig:
                    # Add statistical overlays for all chart types
                    if ((show_ma or show_bollinger) and 
                        pd.api.types.is_numeric_dtype(filtered_data[y_col])):
                        fig = add_statistical_overlays(
                            fig, filtered_data, x_col, y_col, show_ma, ma_window, ma_type,
                            show_bollinger, bollinger_window, bollinger_std
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display filtered data info
                    st.info(f"Showing {len(filtered_data)} of {len(data)} rows")
            
            # Data Normalization Management
            if st.session_state.normalized_columns:
                st.header("📐 Normalized Columns")
                
                # Display existing normalized columns
                for col_name, info in st.session_state.normalized_columns.items():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{col_name}** = {info['numerator']} / {info['denominator']}")
                    with col2:
                        if st.button(f"Remove {col_name}", key=f"remove_{col_name}"):
                            del st.session_state.normalized_columns[col_name]
                            st.rerun()
                
                # Add normalized columns to filtered data for analysis
                for col_name, info in st.session_state.normalized_columns.items():
                    if col_name not in filtered_data.columns:
                        normalized_data = create_normalized_column(filtered_data, info['numerator'], info['denominator'])
                        if normalized_data is not None:
                            filtered_data[col_name] = normalized_data
            
            # Statistical Analysis
            st.header("📊 Statistical Analysis")
            
            # Update numeric columns to include normalized ones
            available_numeric_cols = [col for col in filtered_data.columns if pd.api.types.is_numeric_dtype(filtered_data[col])]
            
            # Select column for analysis
            analysis_col = st.selectbox(
                "Select column for statistical analysis",
                options=available_numeric_cols,
                key="analysis_col"
            )
            
            if analysis_col:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader(f"Statistics for {analysis_col}")
                    display_statistics(filtered_data, analysis_col)
                
                with col2:
                    st.subheader("Distribution")
                    hist_fig = px.histogram(
                        filtered_data, 
                        x=analysis_col, 
                        nbins=30,
                        title=f"Distribution of {analysis_col}"
                    )
                    st.plotly_chart(hist_fig, use_container_width=True)
                

            
            # Data Export
            st.header("💾 Export Data")
            col1, col2 = st.columns(2)
            
            with col1:
                # Export filtered data as CSV
                csv_data = filtered_data.to_csv(index=False)
                st.download_button(
                    label="Download Filtered Data as CSV",
                    data=csv_data,
                    file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Export statistics
                if 'analysis_col' in locals() and analysis_col:
                    stats_data = filtered_data[analysis_col].describe().to_csv()
                    st.download_button(
                        label="Download Statistics as CSV",
                        data=stats_data,
                        file_name=f"statistics_{analysis_col}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    
    else:
        st.info("👆 Please upload a CSV or Excel file to get started!")
        
        # Show example of expected data format
        st.subheader("Expected Data Formats")
        st.write("Your data should be in tabular format with:")
        st.write("- Column headers in the first row")
        st.write("- Each row representing a data point")
        st.write("- Numeric columns for quantitative analysis")
        st.write("- Date/time columns in recognizable formats")
        st.write("- Categorical columns for grouping and filtering")

if __name__ == "__main__":
    main()
