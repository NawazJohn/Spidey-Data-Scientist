import pandas as pd
import sweetviz as sv

# 1. Load the dataset
df = pd.read_csv('dummy_dataset.csv')

# 2. Generate the Auto-EDA report
print("Generating Sweetviz Auto-EDA report. This might take a few seconds...")
report = sv.analyze(df)

# 3. Save it as an HTML file
output_file = "eda_report.html"
report.show_html(output_file, open_browser=False)

print(f"Success! The EDA report has been saved to {output_file}")
