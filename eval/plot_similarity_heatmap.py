import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Find all similarity_*.csv files in the current directory
csv_files = glob.glob('similarity_*.csv')
if not csv_files:
    raise FileNotFoundError('No similarity_*.csv files found!')
frames = []
for file in csv_files:
    frames.append(pd.read_csv(file))
df = pd.concat(frames, ignore_index=True)

# Map model names from filenames/CSV to more readable names for the plot
model_map = {
    'cyclegan_mod_STN': 'CycleGAN (STN+LSGAN)',
    'cyclegan_mod_TPS': 'CycleGAN (TPS+WGAN-GP)',
    'cyclegan_orig': 'CycleGAN (Original)',
    'pix2pix': 'pix2pix'
}
df['model'] = df['model'].map(lambda x: model_map.get(x, x))

#  similarity
pivot = df.pivot_table(index='category', columns='model', values='similarity', aggfunc='mean')

# Sort by category name (index)
pivot = pivot.sort_index()

# Calculate mean similarity across models for each category (for sorting)
category_means = pivot.mean(axis=1)
# Get top 10 lowest and highest mean similarity categories
lowest10 = category_means.nsmallest(10).index.tolist()
highest10 = category_means.nlargest(10).index.tolist()
selected_cats = lowest10 + highest10
subpivot = pivot.loc[selected_cats]

# Transpose for heatmap: categories on x-axis, models on y-axis
subpivot_T = subpivot.transpose()

# Determine figure size based on the number of rows/columns
cell_size = 1.0  # Base size for each cell, adjust as needed
fig_width = max(8, cell_size * len(subpivot_T.columns))
fig_height = max(4, cell_size * len(subpivot_T.index))

plt.close('all')
fig = plt.figure(figsize=(fig_width, fig_height + 1.4))  # Add extra height for title and colorbar
spec = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[18, 1])
ax = fig.add_subplot(spec[0])

sns.set(font_scale=0.95)
heatmap = sns.heatmap(
    subpivot_T,
    annot=True,
    fmt='.3f',
    cmap='YlGnBu',
    linewidths=0.5,
    cbar=False,  # Disable default heatmap colorbar, will add custom one
    ax=ax
)
ax.set_title('Top-10 Lowest & Highest Categories by VGG Feature Cosine Distance', fontsize=14, fontweight='bold', pad=18)
ax.set_xlabel('Category', fontsize=11)
ax.set_ylabel('Model', fontsize=11)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
ax.annotate('(Lower = More Similar)',
            xy=(1, 1.13), xycoords='axes fraction',
            fontsize=10, color='dimgray', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.13', fc='white', ec='none', alpha=0.8))

# Add a subplot for the custom colorbar
cax = fig.add_subplot(spec[1])
# Manually position the colorbar axes: [left, bottom, width, height], adjust as needed
cax.set_position([0.03, 0.04, 0.94, 0.045])
sm = plt.cm.ScalarMappable(cmap='YlGnBu', norm=plt.Normalize(vmin=subpivot_T.min().min(), vmax=subpivot_T.max().max()))
cb = plt.colorbar(sm, cax=cax, orientation='horizontal')
cb.set_label('VGG feature cosine distance (lower is better)', fontsize=10)
cax.xaxis.set_label_position('bottom')
cax.tick_params(labelsize=10)
plt.subplots_adjust(hspace=0.13)  # Adjust space between subplots (heatmap and colorbar)
plt.tight_layout(rect=[0.03, 0.10, 0.97, 0.97])  # Adjust layout to prevent labels from overlapping
plt.savefig('similarity_heatmap_top_bottom10_transposed.png', dpi=100)
print('Saved: similarity_heatmap_top_bottom10_transposed.png')
plt.show()

# Paginate the full heatmap if too many categories, e.g., 30 per page
page_size = 30
categories = list(pivot.index)
num_pages = (len(categories) + page_size - 1) // page_size

for i in range(num_pages):
    subcats = categories[i*page_size:(i+1)*page_size]
    subpivot = pivot.loc[subcats]
    plt.figure(figsize=(min(18, 0.25*subpivot.shape[1]+3), max(6, 0.35*len(subpivot))))
    sns.set(font_scale=0.95)  # Reset font scale for each page
    ax = sns.heatmap(subpivot, annot=True, fmt='.3f', cmap='YlGnBu', linewidths=0.5, cbar_kws={'label': 'Avg. Similarity'})
    plt.title(f'Average Similarity per Category and Model (Page {i+1}/{num_pages})', fontsize=17, fontweight='bold', pad=16)
    plt.ylabel('Category')
    plt.xlabel('Model')
    plt.tight_layout()
    out_path = f'similarity_heatmap_page{i+1}.png'
    plt.savefig(out_path, dpi=100)
    print(f'Saved: {out_path}')
    plt.close()
