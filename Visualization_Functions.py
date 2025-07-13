import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic
import numpy as np

def Plot_Distributions(Class_Distribution, Class_Distribution_Norm, title):
    fig, axs = plt.subplots(1, 2, figsize = (10, 5))

    axs[0].bar(Class_Distribution.index, Class_Distribution.values)
    axs[0].set_xticks(Class_Distribution.index)
    axs[0].set_title('Class Distribution')
    axs[0].set_xlabel('Classes')
    axs[0].set_ylabel('Count')
    axs[1].bar(Class_Distribution_Norm.index, Class_Distribution_Norm.values)
    axs[1].set_xticks(Class_Distribution_Norm.index)
    axs[1].set_title('Class Distribution Normalized')
    axs[1].set_xlabel('Classes')
    axs[1].set_ylabel('Normalized Count')
    
    fig.suptitle(title)

    plt.tight_layout()
    plt.show()

def Plot_Nulls_Heatmap(df):
    cmap = 'Blues'
    fig, axs = plt.subplots(nrows = 3, ncols = 1, figsize = (12, 18))
    sns.heatmap(df.isna().transpose(), cbar = False, cmap = cmap, ax = axs[0])
    axs[0].set_title('Null heatmap for All Passengers')
    sns.heatmap(df[df.Transported == False].isna().transpose(), cbar = False, cmap = cmap, ax = axs[1])
    axs[1].set_title('Null heatmap for Not Transported Passengers')
    sns.heatmap(df[df.Transported == True].isna().transpose(), cbar = False, cmap = cmap, ax = axs[2])
    axs[2].set_title('Null heatmap for Transported Passengers')
    plt.tight_layout()
    plt.show()

def Get_fig_rows_cols(ncols, n):
    if n < ncols:
        ncols = n
        nrows = 1
    elif n % ncols == 0:
        nrows = n//ncols
    else:
        nrows = n//ncols + 1
    return nrows, ncols
    
def Plot_Num_Distributions(df, num_features, target, ncols = 2, bins = 20):
    n = len(num_features)
    nrows, ncols = Get_fig_rows_cols(ncols, n)
    
    plt.subplots(nrows = nrows, ncols = ncols, figsize = (ncols*5, nrows*4))
    x = 1
    for feature in num_features:
        plt.subplot(nrows, ncols, x)
        try:
            sns.histplot(data = df, x = feature, hue = target, kde = True, bins = bins)
        except:
            sns.countplot(data = df, x = feature)
        plt.tight_layout()
        x += 1

def Plot_Stats(df, cat, stats, ncols = 2):
    n_stats = len(stats)
    nrows, ncols = Get_fig_rows_cols(ncols, n_stats)
    fig, axes = plt.subplots(nrows, ncols, figsize = (ncols*9, nrows*6))
    axes = axes.flatten()

    for ax, stat in zip(axes, stats):
        for feature in df.columns.levels[0]:
            sns.lineplot(
                x = df.index,
                y = df[(feature, stat)],
                marker = 'o',
                ax = ax,
                label = feature
            )
        ax.set_title(f'{stat} by {cat}')
        ax.set_xlabel(cat)
        ax.set_ylabel(stat)
        if stat != stats[0]:
            ax.legend_.remove()
    
    plt.tight_layout()
    plt.show()

def Get_Mosaic(df, features, target):
    n = len(features)
    ncols = 2
    if n < ncols:
        ncols = n
        nrows = 1
    elif n % ncols == 0:
        nrows = n // ncols
    else:
        nrows = n // ncols + 1

    fig, axes = plt.subplots(nrows, ncols, figsize = (ncols*9, nrows*6))
    axes = axes.flatten()

    for ax, feature in zip(axes, features):
        mosaic(df, [feature, target], ax = ax)
        ax.set_title(feature)
        
    plt.tight_layout()
    plt.show()

def Gridplot_Hists(df_orig, features, max_cats = 5):
    df = df_orig[features].dropna().copy()
    df[features] = df[features].astype(str)

    for feature in ['Age_Bin', 'FamilySize', 'FirstNameLength', 'LastNameLength']:
        if feature in features:
            df[feature] = df[feature].astype(int)

    n = len(features)

    fig, axes = plt.subplots(nrows = n, ncols = n, figsize = (5*n, 5*n))

    for i, c1 in enumerate(features):
        for j, c2 in enumerate(features):
            ax = axes[i, j]
            if i == j:
                sns.histplot(df, x = c1, stat = 'proportion', ax = ax)
            else:
                sns.histplot(df, x = c1, stat = 'density', hue = c2, multiple = 'fill', ax = ax)

            cats = df[c1].dropna().unique().astype(str)
            if len(cats) > max_cats and any(len(lbl) > 2 for lbl in cats):
                ax.tick_params(axis = 'x', rotation = 45)

    plt.tight_layout()
    plt.show()

def Plot_MI_Heatmap(MI, MI_min, MI_max):
    plt.figure(figsize = (10, 8))
    sns.heatmap(MI, annot = True, fmt = ".2f", cmap = "coolwarm", vmin = MI_min, vmax = MI_max)
    plt.xticks(rotation = 45)
    plt.show()

def Plot_MI(MI, topk = 10, bottom = True):
    top_features = MI[:topk]
    
    if bottom:
        mask = MI != 0.0
        bottom_features = MI[mask][-topk:]
    
        fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (9, 12))
    
        sns.barplot(x = top_features.values, y = top_features.index, orient = 'h', ax = axes[0])
        axes[0].set_xlabel('Mutual Information')
        axes[0].set_title(f'MI values of Top {topk} Features')
        
        sns.barplot(x = bottom_features.values, y = bottom_features.index, orient = 'h', ax = axes[1])
        axes[1].set_xlabel('Mutual Information')
        axes[1].set_title(f'MI values of Bottom {topk} Non-Zero Features')
    else:
        plt.figure(figsize = (9, 6))
        ax = sns.barplot(x = top_features.values, y = top_features.index, orient = 'h')
        ax.set_xlabel('Mutual Information')
        ax.set_title(f'MI values of Top {topk} Features')
        
    plt.tight_layout()
    plt.show()

def PairGrid_kde(df, num_features, hue = None):
    if hue is None:
        g = sns.PairGrid(df, vars = num_features)
        g.map_lower(sns.kdeplot,
                    levels = 20,
                    fill = True,
                    thresh = 0.1,
                    cmap = "vlag")
        g.map_upper(sns.kdeplot,
                    levels = 20,
                    fill = True,
                    thresh = 0.1,
                    cmap = "vlag")
    else:
        g = sns.PairGrid(df, vars = num_features, hue = hue)
        g.map_lower(sns.kdeplot, 
            levels = 20, 
            fill = True, 
            alpha = 0.5, 
            thresh = 0.1)
        g.map_upper(sns.kdeplot,
            levels = 20, 
            fill = True, 
            thresh = 0.5, 
            alpha = 0.5)
    g.map_diag(sns.kdeplot)

def plot_corrs(X, cmap = 'coolwarm'):
    d2 = X.shape[1]
    d1 = d2 + 8
    correlations = X.corr()
    mask = np.zeros_like(correlations, dtype = 'bool')
    mask[np.triu_indices_from(mask)] = True
    fig, ax = plt.subplots(figsize = (d1, d2))
    sns.heatmap(correlations, cmap = cmap, mask = mask)
    plt.show()

def get_ecdfplots(df, variables, target, ncols):
    n = len(variables)
    nrows = n//ncols + 1
    plt.figure(figsize = (ncols*3, nrows*3))
    
    x = 1
    for feature in variables:
        plt.subplot(nrows, ncols, x)
        if x == 1:
            sns.ecdfplot(data = df, x = feature, hue = target)
        else:
            sns.ecdfplot(data = df, x = feature, hue = target, legend = False)
        plt.tight_layout()
        x += 1

def get_violinplots(df, variables, target1, ncols, target2 = None):
    n = len(variables)
    
    if (n == 1) & (target2 != None):
        feature = variables[0]
        plt.figure(figsize = (10, 5))
        plt.subplot(1, 2, 1)
        sns.violinplot(data = df, x = target1, y = feature, hue = target2, split = True)
        plt.subplot(1, 2, 2)
        sns.violinplot(data = df, x = target2, y = feature, hue = target1, split = True)
        return
    
    nrows = n//ncols + 1
    if ncols < 3:
        plt.figure(figsize = (ncols*5, nrows*5))
    else:
        plt.figure(figsize = (ncols*3, nrows*3))
    
    x = 1
    for feature in variables:
        plt.subplot(nrows, ncols, x)
        if target2 == None:
            sns.violinplot(data = df, x = target1, y = feature, hue = target1, split = False)
        else:
            sns.violinplot(data = df, x = target1, y = feature, hue = target2, split = True)
        plt.tight_layout()
        x += 1

def plot_hists_by_Cryo_and_Transported(df, feature, bins = 10):
    plt.figure(figsize = (20, 20))
    plt.subplot(4, 1, 1)
    sns.histplot(data = df, x = feature, bins = bins, multiple = 'layer', hue = 'Transported')
    plt.title(f'{feature} Distribution')
    plt.subplot(4, 1, 2)
    sns.histplot(data = df, x = feature, bins = bins, multiple = 'layer', hue = 'CryoSleep')
    plt.subplot(4, 1, 3)
    sns.histplot(data = df, x = feature, bins = 2*bins, multiple = 'layer', hue = 'Transported')
    plt.subplot(4, 1, 4)
    sns.histplot(data = df, x = feature, bins = 2*bins, multiple = 'layer', hue = 'CryoSleep')

def plot_ratios_by_Cryo_and_Transported(df, feature, nbins = None, cat = False, cat_bins = None):
    df_num = df[['CryoSleep', feature, 'Transported']].copy()
    if cat:
        group = feature
        bins = np.sort(df_num[feature].unique()).tolist() if cat_bins == None else cat_bins
    else:
        assert(nbins != None), f"nbins needs to be given if feature is not categorical. Got {nbins = }, {cat = }"
        group = 'bins'
        counts, bin_edges = np.histogram(df[feature], bins = nbins)
        precision = 1
        bins = [round(bin_edges[i], precision) for i in range(len(bin_edges))]
        df_num['bins'] = np.digitize(df_num[feature], bins = bins, right = False)
        df_num['bins'] = np.where(df_num['bins'] == nbins + 1, nbins, df_num['bins'])
    bin_ranges = [(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]
        
    mask_T = df_num['Transported'] == 1
    mask_C = df_num['CryoSleep'] == 1
    df_num['Transported in CryoSleep'] = np.where(mask_T & mask_C, 1, 0)
    df_num['Transported not in CryoSleep'] = np.where(mask_T & ~mask_C, 1, 0)

    df_num_groups = df_num.groupby(group).agg(
        **{'Total Count': (feature, 'count'), 
           'Total in CryoSleep': ('CryoSleep', 'sum'), 
           'Total Transported': ('Transported', 'sum'), 
           'Total Transported in CryoSleep': ('Transported in CryoSleep', 'sum'), 
           'Total Transported not in CryoSleep': ('Transported not in CryoSleep', 'sum')}
    )

    df_num_groups['Ratio Transported'] = df_num_groups['Total Transported']/df_num_groups['Total Count']
    df_num_groups['Ratio in CryoSleep'] = df_num_groups['Total in CryoSleep']/df_num_groups['Total Count']
    df_num_groups['Ratio Transported in CryoSleep'] = (
        df_num_groups['Total Transported in CryoSleep']/df_num_groups['Total in CryoSleep'])
    df_num_groups['Ratio Transported not in CryoSleep'] = (
        df_num_groups['Total Transported not in CryoSleep']/
        (df_num_groups['Total Count'] - df_num_groups['Total in CryoSleep']))
    
    g_features = ['Ratio Transported', 'Ratio Transported in CryoSleep', 'Ratio Transported not in CryoSleep']
    plt.figure(figsize = (20, 20))
    for k in range(3):
        plt.subplot(3, 1, k + 1)
        if cat:
            data = df_num_groups[df_num_groups[g_features[k]].notnull()]
            sns.barplot(data, x = data.index, y = g_features[k], order = bins,
                        palette = plt.get_cmap('jet'), hue = g_features[k], legend = False)
        else:
           sns.barplot(df_num_groups, x = df_num_groups.index, y = g_features[k],
                        palette = plt.get_cmap('jet'), hue = g_features[k], legend = False)

        
    return bin_ranges, df_num_groups

def plot_CryoTrans_Counts(df_orig, x, order):
    required_cols = [x, 'CryoSleep', 'Transported']
    missing = [c for c in required_cols if c not in df_orig.columns]
    assert not missing, f"Missing required columns: {missing}"
    
    df = df_orig.copy()
    df.CryoSleep = df.CryoSleep.astype(bool).astype(str)
    df.Transported = df.Transported.astype(bool).astype(str)
    df['Cryo_Trans'] = df['CryoSleep'] + '_' + df['Transported']
    hue_order = ['False_False', 'False_True', 'True_False', 'True_True']

    plt.figure(figsize = (12, 12))

    plt.subplot(2, 2, 1)
    sns.countplot(x = x, data = df, order = order)
    plt.xticks(rotation = 45)

    plt.subplot(2, 2, 2)
    sns.countplot(x = x, hue = 'Transported', data = df, order = order, hue_order = ['False', 'True'])
    plt.xticks(rotation = 45)
    plt.ylabel('')

    plt.subplot(2, 2, 3)
    sns.countplot(x = x, hue = 'CryoSleep', data = df, order = order, hue_order =  ['False', 'True'])
    plt.xticks(rotation = 45)

    plt.subplot(2, 2, 4)
    sns.countplot(x = x, hue = 'Cryo_Trans', data = df, order = order, hue_order = hue_order)
    plt.xticks(rotation = 45)
    plt.ylabel('')

    plt.tight_layout()
    plt.show()