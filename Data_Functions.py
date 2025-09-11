import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import numpy as np
import string
from sklearn.impute import KNNImputer
from sklearn.preprocessing import  RobustScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LassoCV
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold

def Get_Data(filename):
    data = pd.read_csv(filename)
    return data

def Explore_Data(df):
    print('shape:\n', df.shape, '\n')
    print('column names:\n', df.columns, '\n')

    print('info:\n', df.info(), '\n')
    print('summary statistics: \n', df.describe(), '\n')
    print('number of missing values: \n', df.isnull().sum(), '\n')
    
def Get_Class_Distribution(df, target):
    classes = df[target].unique()
    distr = df[target].value_counts()
    distr_norm = df[target].value_counts(normalize = True)

    return classes, distr, distr_norm

def Get_MI_Matrices(df_orig, Features, target, classif = True):
    assert(df_orig[target].dtype == 'bool'), "Error in Get_MI_Matrices: Expecting a boolean binary target"
    df = df_orig[Features + [target]].dropna().copy()
    MI_matrix = pd.DataFrame(np.zeros((len(Features), len(Features))), index = Features, columns = Features)
    MI_matrix_T = pd.DataFrame(np.zeros((len(Features), len(Features))), index = Features, columns = Features)
    MI_matrix_NT = pd.DataFrame(np.zeros((len(Features), len(Features))), index = Features, columns = Features)

    if classif:
        mutual_info_fn = mutual_info_classif
    else:
        mutual_info_fn = mutual_info_regression

    for feature in Features:
        X = df.drop(columns = [target])
        X_T = df[df[target] == True].drop(columns = [target])
        X_NT = df[df[target] == False].drop(columns = [target])
        y = df[feature]
        y_T = df[df[target] == True][feature]
        y_NT = df[df[target] == False][feature]
   
        MI = pd.Series(mutual_info_fn(X, y), index = X.columns.tolist())
        MI_T = pd.Series(mutual_info_fn(X_T, y_T), index = X_T.columns.tolist())
        MI_NT = pd.Series(mutual_info_fn(X_NT, y_NT), index = X_NT.columns.tolist())
        MI_matrix.loc[feature] = MI
        MI_matrix_T.loc[feature] = MI_T
        MI_matrix_NT.loc[feature] = MI_NT

    np.fill_diagonal(MI_matrix.values, 0)
    np.fill_diagonal(MI_matrix_T.values, 0)
    np.fill_diagonal(MI_matrix_NT.values, 0)

    MI_max = max(MI_matrix.values.max(), MI_matrix_T.values.max(), MI_matrix_NT.values.max())
    MI_min = 0

    return MI_matrix, MI_matrix_T, MI_matrix_NT, MI_min, MI_max

def Extract_PassengerId_Info(df):
    df2 = df.copy()
    df2_split = df2.PassengerId.str.split("_", expand = True).rename({0: 'GroupId', 1: 'ppId'}, axis = 1)
    df2 = pd.concat([df2, df2_split], axis = 1)
    df2['GroupSize'] = df2.groupby('GroupId')['ppId'].transform(len)
    df2['ppId'] = df2['ppId'].astype(int)

#     df2['ppId'] = LabelEncoder().fit_transform(df2['ppId'])
#     df2['PassengerInfo'] = (8*df2['ppId'] + df2['GroupSize'] - 1 - (df2['ppId'] * (1 + df2['ppId']))/2).astype(int)
    
    df2_split = df2.GroupId.str.extractall('(.)')[0].unstack().rename({0: 'G1', 1: 'G2', 2: 'G3', 3: 'G4'}, axis = 1)
    df2 = pd.concat([df2, df2_split], axis = 1)
    
    return df2

def LetterCount(string, letter):
    return string.lower().count(letter)

def Extract_Initial_Data_orig(df):
    df = Extract_PassengerId_Info(df)
    df_split = df.Cabin.str.split("/", expand = True).rename({0: 'deck', 1: 'Cabin Number', 2: 'side'}, axis=1)
    df = pd.concat([df, df_split], axis = 1)
    df_split = df.Name.str.split(" ", expand = True).rename({0: 'First Name', 1: 'Last Name'}, axis=1)
    df = pd.concat([df, df_split], axis = 1)
    df['FamilySize'] = df.groupby('Last Name')['Last Name'].transform(len)
    df['FirstNameLength'] = df['First Name'].fillna('').astype(str).apply(len).replace(0, np.nan)
    df['LastNameLength'] = df['Last Name'].fillna('').astype(str).apply(len).replace(0, np.nan)
    df['GroupFamilySize'] = df.groupby(['Last Name', 'GroupId'])['Last Name'].transform(len)
    df['CabinFamilySize'] = df.groupby(['Last Name', 'Cabin'])['Cabin'].transform(len)
    df['CabinGroupSize'] = df.groupby(['GroupId', 'Cabin'])['Cabin'].transform(len)
    df['CabinSize'] = df.groupby('Cabin')['Cabin'].transform(len)
    
    for Letter in list(string.ascii_lowercase):
        df[Letter] = df['Name'].fillna('').apply(LetterCount, args = (Letter))
        mask = df['Name'].isnull()
        df.loc[mask, Letter] = df.loc[mask, Letter].replace(0, np.nan)
    
    df.set_index('PassengerId', inplace = True)
    
    return df

def Extract_Initial_Data(df_orig, df_test_orig, version = 1, regions_bin_edges = [0, 316, 758, 1137, 1516], bin_edges = [1, 3713, 4641, 6497, 7425]):
    df = df_orig.copy()
    df_test = df_test_orig.copy()
    df_test['Transported'] = np.nan
    df_all = pd.concat([df, df_test]).reset_index(drop = True)
    df_all = Extract_Initial_Data_orig(df_all)
    
    if version == 1:
        df_all.drop(columns = ['Cabin', 'Name', 'First Name'], inplace = True)
    elif version == 2:
        df_all['GroupId'] = df_all['GroupId'].astype(int)
        df_all['Cabin Number'] = df_all['Cabin Number'].astype(pd.Int64Dtype())
        df_all = Get_Regions(df_all, 'GroupId', 'Batch', bin_edges)
        df_all = Get_Regions(df_all, 'Cabin Number', 'Region', regions_bin_edges)
        df_all['Under_13'] = df_all.Age.where(df_all.Age.isnull(), df_all.Age < 13)
        bin_edges = [0, 2, 5, 13, 20, 36, 46, 65, 90]
        LifeStages_choices = ['Infant', 'Toddler', 'Child', 'Teen', 'Young Adult', 'Young middle aged', 'Older middle aged', 'Senior']
        df_all['Age_LifeStages_lbs'] = pd.cut(df_all.Age, bins = bin_edges, labels = LifeStages_choices, right = False, include_lowest = True)
        df_all['Age_LifeStages'] = df_all.Age_LifeStages_lbs.cat.codes.replace(-1, np.nan)
        df_all.drop(columns = list(string.ascii_lowercase) + 
                    ['G1', 'G2', 'G3', 'G4', 'Name', 'First Name'], 
                    inplace = True)
        
    df = df_all.iloc[: len(df)]
    df_test = df_all.iloc[-len(df_test) :]
    return df, df_test.drop(columns = ['Transported'])

def fill_NA(df, mask, col, value):
    with pd.option_context('future.no_silent_downcasting', True):
        df.loc[mask, col] = df.loc[mask, col].fillna(value).infer_objects(copy = False)
    return df

def Clean_Data(df_orig):
    df = df_orig.copy()
    
    # replace nan HomePlanet with that of their family members if available
    
    Family_to_HomePlanet = (df.dropna(subset = ['HomePlanet'])
                            .groupby('Last Name')['HomePlanet']
                            .agg(lambda x: x.mode()[0] if not x.empty else None).to_dict())
    
    def fill_HomePlanet(row):
        if pd.isnull(row['HomePlanet']):
            return Family_to_HomePlanet.get(row['Last Name'], row['HomePlanet'])
        return row['HomePlanet']
    
    df['HomePlanet'] = df.apply(fill_HomePlanet, axis = 1)

    # replace nan HomePlanet with that of their group members if available
    
    Group_to_HomePlanet = (df.dropna(subset = ['HomePlanet'])
                           .groupby('GroupId')['HomePlanet']
                           .agg(lambda x: x.mode()[0] if not x.empty else None).to_dict())
    
    def fill_HomePlanet_wgroup(row):
        if pd.isnull(row['HomePlanet']):
            return Group_to_HomePlanet.get(row['GroupId'], row['HomePlanet'])
        return row['HomePlanet']
    
    df['HomePlanet'] = df.apply(fill_HomePlanet_wgroup, axis = 1)
    
    # replace all nan luxury features by zero if age is less than 13
    luxury_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    mask = df['Age'] < 13
    for feature in luxury_features:
        df.loc[mask, feature] = df.loc[mask, feature].fillna(0)
        
    # replace all nan luxury features by zero if CryoSleep = True
    mask = df['CryoSleep'] == True
    for feature in luxury_features:
        df.loc[mask, feature] = df.loc[mask, feature].fillna(0)
    
    # replace all nan VIP with False if Age < 18
    mask = df['Age'] < 18
    df = fill_NA(df, mask, 'VIP', False)
    # df.loc[mask, 'VIP'] = df.loc[mask, 'VIP'].fillna(False)
    
    # replace all nan VIP with False if HomePlanet is Earth
    mask = df['HomePlanet'] == 'Earth'
    df = fill_NA(df, mask, 'VIP', False)
    
    # replace all nan VIP with False if HomePlanet is Mars and Destination is 55 Cancri e
    mask = (df['HomePlanet'] == 'Mars') & (df['Destination'] == '55 Cancri e')
    df = fill_NA(df, mask, 'VIP', False)
    
    # replace all nan VIP with False if deck is G or T
    mask = (df['deck'] == 'G') | (df['deck'] == 'T')
    df = fill_NA(df, mask, 'VIP', False)
    
    # Obviously if any of the luxury features is not zero then CryoSleep must be False
    mask = df[luxury_features].sum(axis = 1) > 0
    df = fill_NA(df, mask, 'CryoSleep', False)
    
    # additionally
    mask1 = ((df['RoomService'] == 0) & (df['FoodCourt'] == 0) & 
         (df['ShoppingMall'] == 0) & (df['Spa'] == 0) & (df['VRDeck'] == 0))
    mask2 = ((df['Destination'] == 'PSO J318.5-22') | (df['Destination'] == 'Cancri e'))
    mask3 = df['Age'] > 12
    mask = mask1 & mask2 & mask3
    df = fill_NA(df, mask, 'CryoSleep', True)
    
    # Passengers on decks A, B, C, or T are from Europa
    mask = (df['deck'] == 'A') | (df['deck'] == 'B') | (df['deck'] == 'C') | (df['deck'] == 'T')
    df.loc[mask, 'HomePlanet'] = df.loc[mask, 'HomePlanet'].fillna('Europa')
    
    # Passengers on deck G are from Earth
    mask = df['deck'] == 'G'
    df.loc[mask, 'HomePlanet'] = df.loc[mask, 'HomePlanet'].fillna('Earth')
    
    # If deck is D and Destination is PSO J318.5-22 then HomePlanet is Mars
    mask = (df['deck'] == 'D') & (df['Destination'] == 'PSO J318.5-22')
    df.loc[mask, 'HomePlanet'] = df.loc[mask, 'HomePlanet'].fillna('Mars')
    # df.loc[mask, 'CryoSleep'] = df.loc[mask, 'CryoSleep'].fillna(False)
    df = fill_NA(df, mask, 'CryoSleep', False)
    
    # If deck is F and VIP is True then HomePlanet is Mars
    mask = (df['deck'] == 'F') & (df['VIP'] == True)
    df.loc[mask, 'HomePlanet'] = df.loc[mask, 'HomePlanet'].fillna('Mars')
    
    # If FamilySize > 16 then HomePlanet is Earth
    mask = df['FamilySize'] > 16
    df.loc[mask, 'HomePlanet'] = df.loc[mask, 'HomePlanet'].fillna('Earth')

    # Passengers with a long first name are from Europa
    mask = df['FirstNameLength'] > 6
    df.loc[mask, 'HomePlanet'] = df.loc[mask, 'HomePlanet'].fillna('Europa')

    # Passengers with a short last name are from Mars
    mask = df['LastNameLength'] < 5
    df.loc[mask, 'HomePlanet'] = df.loc[mask, 'HomePlanet'].fillna('Mars')
    
    # If FamilySize = 1 then obviously CabinFamilySize = 1
    mask = df['FamilySize'] == 1
    df.loc[mask, 'CabinFamilySize'] = df.loc[mask, 'CabinFamilySize'].fillna(1)
    
    # If GroupSize = 1 then CabinGroupSize = 1
    mask = df['GroupSize'] == 1
    df.loc[mask, 'CabinSize'] = df.loc[mask, 'CabinSize'].fillna(1)
    df.loc[mask, 'CabinFamilySize'] = df.loc[mask, 'CabinFamilySize'].fillna(1)
    df.loc[mask, 'GroupFamilySize'] = df.loc[mask, 'GroupFamilySize'].fillna(1)

    return df

def Clean_and_dropna(df_orig, df_test_orig):
    df = df_orig.copy()
    df_test = df_test_orig.copy()
    df_test['Transported'] = np.nan
    df_all = pd.concat([df, df_test])
    
    null_mask = df_all.drop(columns = ['Transported']).isnull().any(axis = 1)
    print('Number of null rows before cleaning: ', len(df_all[null_mask]))    
    df_clean = Clean_Data(df_all)

    numerical_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Age', 'GroupSize', 
                          'FamilySize', 'FirstNameLength', 'LastNameLength', 'GroupFamilySize', 
                          'CabinFamilySize', 'CabinGroupSize', 'CabinSize']
    
    skewness = df_clean[numerical_features].skew().apply(np.abs)
    skewed_features = skewness[skewness > 0.6].index.to_list()
    print('skewed_features: ', skewed_features)
    df_clean[skewed_features] = df_clean[skewed_features].apply(np.log1p)

    df = df_clean.iloc[: len(df)].drop(columns = ['Last Name'])
    df_test = (df_clean.iloc[-len(df_test) :]).drop(columns = ['Transported', 'Last Name'])
    
    null_mask = df.isnull().any(axis = 1)
    print(f'drop remaining {len(df[null_mask])} null rows in training data')
    df_clean = df.dropna(axis = 0, how = 'any').copy()
    
    null_mask = df_test.isnull().any(axis = 1)
    print(f'drop remaining {len(df_test[null_mask])} null rows in test data')
    df_test_clean = df_test.dropna(axis = 0, how = 'any').copy()

    return df_clean, df_test_clean

def Clean_and_preImpute(df_orig, df_test_orig):
    df = df_orig.copy()
    df_test = df_test_orig.copy()
    df_test['Transported'] = np.nan
    df_all = pd.concat([df, df_test])

    print(f'Number of null rows before cleaning: {len(df_orig[df_orig.isnull().any(axis = 1)])} in training data\
    and {len(df_test_orig[df_test_orig.isnull().any(axis = 1)])} in testing data')
    
    df_clean = Clean_Data(df_all)

    nominal_features = ['HomePlanet', 'Destination', 'side', 'deck']
    binary_features = ['CryoSleep', 'VIP', 'Batch_1', 'Batch_2', 'Batch_3', 'Batch_4', 'Batch_5', 
                       'Region_1', 'Region_2', 'Region_3', 'Region_4', 'Region_5', 'Under_13']

    numerical_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Age', 'GroupSize', 
                          'FamilySize', 'FirstNameLength', 'LastNameLength', 'GroupFamilySize', 
                          'CabinFamilySize', 'CabinGroupSize', 'CabinSize', 'ppId']
    
    skewness = df_clean[numerical_features].skew().apply(np.abs)
    skewed_features = skewness[skewness > 0.6].index.to_list()
    print('skewed_features: ', skewed_features)
    df_clean[skewed_features] = df_clean[skewed_features].apply(np.log1p)

    df_clean = Pre_Prep_Data(df_clean, nominal_features, binary_features)
    df_clean = Pre_Impute_Features(df_clean)

    df = df_clean.iloc[: len(df)]
    df_test = (df_clean.iloc[-len(df_test) :]).drop(columns = ['Transported'])
    
    null_mask = df.isnull().any(axis = 1)
    print(f'There are {len(df[null_mask])} remaining null rows in training data after cleaning')
    
    null_mask = df_test.isnull().any(axis = 1)
    print(f'and {len(df_test[null_mask])} remaining null rows in test data')
    
    return df, df_test

def Get_Regions(df, feature, Region, bins_edges):
    Regions = []
    nbins = len(bins_edges)

    Region_null = f'{Region}_0'
    if df[feature].isnull().any():
        df[Region_null] = df[feature].isnull().astype(int)
        Regions.append(Region_null)
    
    for k in range(nbins):
        Region_name = f'{Region}_{k + 1}'
        Regions.append(Region_name)
        if k == nbins - 1:
            df[Region_name] = df[feature] >= bins_edges[k]
        else:
            df[Region_name] = (df[feature] >= bins_edges[k]) & (df[feature] < bins_edges[k + 1])

        df[Region_name] = df[Region_name].fillna(False).astype(int)

    df[Region] = df[Regions].idxmax(axis = 1)
    df[Region] = df[Region].str.replace(f'{Region}_', '').astype(int)

    if Region_null in Regions:
        df[Region] = df[Region].replace(0, np.nan)
        Regions.remove(Region_null)
        df.drop(Region_null, axis = 1, inplace = True)
        mask = df[Region].isnull()
        for region in Regions:
            df.loc[mask, region] = df.loc[mask, region].replace(0, np.nan)

    return df

def Pre_Prep_Data(df_orig, nominal_features, binary_features):
    df = df_orig.copy()
    
    for feature in nominal_features:
        one_hot_encoded = pd.get_dummies(df[feature], prefix = feature)
        mask = (~one_hot_encoded).all(axis = 1)
        one_hot_encoded = one_hot_encoded.astype(int)
        one_hot_encoded[mask] = np.nan
        df = pd.concat([df, one_hot_encoded], axis = 1)
        
    for feature in binary_features:
        col = df[feature].copy()
        mask = col.isnull()
        col[~mask] = col[~mask].astype(int)
        df[feature] = col
            
    return df

def Pre_Impute_Features(df_orig):
    df = df_orig.copy()
    
    mask = (df['deck_A'] == 1) | (df['deck_B'] == 1) | (df['deck_C'] == 1) | (df['deck_D'] == 1) | (df['deck_T'] == 1)
    df.loc[mask, 'HomePlanet_Earth'] = df.loc[mask, 'HomePlanet_Earth'].fillna(0)
    
    mask = (df['deck_F'] == 1) | (df['deck_G'] == 1)
    df.loc[mask, 'HomePlanet_Europa'] = df.loc[mask, 'HomePlanet_Europa'].fillna(0)
    
    mask = df['Destination_PSO J318.5-22'] == 1
    df.loc[mask, 'deck_T'] = df.loc[mask, 'deck_T'].fillna(0)
    
    mask = df['VIP'] == 1
    df.loc[mask, 'HomePlanet_Earth'] = df.loc[mask, 'HomePlanet_Earth'].fillna(0)
    # df.loc[mask, 'Under_13'] = df.loc[mask, 'Under_13'].fillna(0)
    df = fill_NA(df, mask, 'Under_13', 0)
    
    mask = df['FirstNameLength'] < 4
    df.loc[mask, 'HomePlanet_Europa'] = df.loc[mask, 'HomePlanet_Europa'].fillna(0)
    
    mask = df['FirstNameLength'] > 6
    df.loc[mask, 'HomePlanet_Europa'] = df.loc[mask, 'HomePlanet_Europa'].fillna(1)
    df.loc[mask, 'HomePlanet_Earth'] = df.loc[mask, 'HomePlanet_Earth'].fillna(0)
    df.loc[mask, 'HomePlanet_Mars'] = df.loc[mask, 'HomePlanet_Mars'].fillna(0)
    
    mask = df['LastNameLength'] < 5
    df.loc[mask, 'HomePlanet_Europa'] = df.loc[mask, 'HomePlanet_Europa'].fillna(0)
    df.loc[mask, 'HomePlanet_Earth'] = df.loc[mask, 'HomePlanet_Earth'].fillna(0)
    df.loc[mask, 'HomePlanet_Mars'] = df.loc[mask, 'HomePlanet_Mars'].fillna(1)
    
    mask = df['LastNameLength'] > 5
    df.loc[mask, 'HomePlanet_Mars'] = df.loc[mask, 'HomePlanet_Mars'].fillna(0)
    
    mask = df['LastNameLength'] < 6
    df.loc[mask, 'HomePlanet_Europa'] = df.loc[mask, 'HomePlanet_Europa'].fillna(0)
    
    mask = df['HomePlanet_Earth'] == 1
    df.loc[mask, 'deck_A'] = df.loc[mask, 'deck_A'].fillna(0)
    df.loc[mask, 'deck_B'] = df.loc[mask, 'deck_B'].fillna(0)
    df.loc[mask, 'deck_C'] = df.loc[mask, 'deck_C'].fillna(0)
    df.loc[mask, 'deck_D'] = df.loc[mask, 'deck_D'].fillna(0)
    df.loc[mask, 'deck_T'] = df.loc[mask, 'deck_T'].fillna(0)
    df.loc[mask, 'HomePlanet'] = df.loc[mask, 'HomePlanet'].fillna('Earth')
    
    mask = df['HomePlanet_Europa'] == 1
    df.loc[mask, 'deck_F'] = df.loc[mask, 'deck_F'].fillna(0)
    df.loc[mask, 'deck_G'] = df.loc[mask, 'deck_G'].fillna(0)
    df.loc[mask, 'HomePlanet'] = df.loc[mask, 'HomePlanet'].fillna('Europa')
    
    mask = df['HomePlanet_Mars'] == 1
    df.loc[mask, 'deck_A'] = df.loc[mask, 'deck_A'].fillna(0)
    df.loc[mask, 'deck_B'] = df.loc[mask, 'deck_B'].fillna(0)
    df.loc[mask, 'deck_C'] = df.loc[mask, 'deck_C'].fillna(0)
    df.loc[mask, 'deck_G'] = df.loc[mask, 'deck_G'].fillna(0)
    df.loc[mask, 'deck_T'] = df.loc[mask, 'deck_T'].fillna(0)
    df.loc[mask, 'HomePlanet'] = df.loc[mask, 'HomePlanet'].fillna('Mars')
    
    mask = df['VIP'] == 1
    df.loc[mask, 'deck_G'] = df.loc[mask, 'deck_G'].fillna(0)
    df.loc[mask, 'deck_T'] = df.loc[mask, 'deck_T'].fillna(0)
        
    if df['GroupSize'].max() < np.log1p(9):
        mask = df['GroupSize'] > np.log1p(3)
        df.loc[mask, 'deck_T'] = df.loc[mask, 'deck_T'].fillna(0)
        
        mask = df['GroupSize'] > np.log1p(6)
        df.loc[mask, 'deck_A'] = df.loc[mask, 'deck_A'].fillna(0)
        
        mask = df['GroupSize'] > np.log1p(7)
        df.loc[mask, 'deck_B'] = df.loc[mask, 'deck_B'].fillna(0)
        df.loc[mask, 'deck_C'] = df.loc[mask, 'deck_C'].fillna(0)
        df.loc[mask, 'deck_D'] = df.loc[mask, 'deck_D'].fillna(0)
        df.loc[mask, 'deck_E'] = df.loc[mask, 'deck_E'].fillna(0)
        
    mask = df['HomePlanet_Europa'] == 1
    df.loc[mask, 'Region_3'] = df.loc[mask, 'Region_3'].fillna(0)
    df.loc[mask, 'Region_4'] = df.loc[mask, 'Region_4'].fillna(0)
    df.loc[mask, 'Region_5'] = df.loc[mask, 'Region_5'].fillna(0)
        
    mask2 = mask & (df['GroupSize'] > np.log1p(6))
    df.loc[mask2, 'Region'] = df.loc[mask2, 'Region'].fillna(1)
    df.loc[mask2, 'Region_1'] = df.loc[mask2, 'Region_1'].fillna(1)
    df.loc[mask2, 'Region_2'] = df.loc[mask2, 'Region_2'].fillna(0)
    df.loc[mask2, 'Region_3'] = df.loc[mask2, 'Region_3'].fillna(0)
    df.loc[mask2, 'Region_4'] = df.loc[mask2, 'Region_4'].fillna(0)
    df.loc[mask2, 'Region_5'] = df.loc[mask2, 'Region_5'].fillna(0)
        
    mask2 = mask & (df['Batch'] < 3)
    df.loc[mask2, 'Region'] = df.loc[mask2, 'Region'].fillna(1)
    df.loc[mask2, 'Region_1'] = df.loc[mask2, 'Region_1'].fillna(1)
    df.loc[mask2, 'Region_2'] = df.loc[mask2, 'Region_2'].fillna(0)
    df.loc[mask2, 'Region_3'] = df.loc[mask2, 'Region_3'].fillna(0)
    df.loc[mask2, 'Region_4'] = df.loc[mask2, 'Region_4'].fillna(0)
    df.loc[mask2, 'Region_5'] = df.loc[mask2, 'Region_5'].fillna(0)
        
    mask = df['Batch'] < 3
    df.loc[mask, 'Region_4'] = df.loc[mask, 'Region_4'].fillna(0)
        
    mask = df['Batch'] < 4
    df.loc[mask, 'Region_5'] = df.loc[mask, 'Region_5'].fillna(0)
        
    mask = (df['HomePlanet_Earth'] == 1) & (df['Batch'] == 5)
    df.loc[mask, 'Region_1'] = df.loc[mask, 'Region_1'].fillna(0)
    df.loc[mask, 'Region_3'] = df.loc[mask, 'Region_3'].fillna(0)
        
    mask = (df['HomePlanet_Earth'] == 1) & (df['Batch'] == 4)
    df.loc[mask, 'Region_1'] = df.loc[mask, 'Region_1'].fillna(0)
        
    mask = (df['HomePlanet_Mars'] == 1) & (df['Batch'] > 3)
    df.loc[mask, 'Region_3'] = df.loc[mask, 'Region_3'].fillna(0)
        
    mask = df['Region'] > 2
    df.loc[mask, 'HomePlanet_Europa'] = df.loc[mask, 'HomePlanet_Europa'].fillna(0)
        
    mask = (df['HomePlanet_Europa'] == 1) & (df['Region_2'] == 1) & (df['Batch'] < 5)
    df.loc[mask, 'Destination_PSO J318.5-22'] = df.loc[mask, 'Destination_PSO J318.5-22'].fillna(0)
        
    mask = (df['Batch_1'] == 1) & (df['Region_3'] == 1)
    df.loc[mask, 'Destination_PSO J318.5-22'] = df.loc[mask, 'Destination_PSO J318.5-22'].fillna(0)
        
    mask = (df['Batch_1'] == 1) & (df['Destination_PSO J318.5-22'] == 1)
    df.loc[mask, 'Region_3'] = df.loc[mask, 'Region_3'].fillna(0)
        
    def Family_to_planet(planet):
        return (df.dropna(subset = [planet]).groupby('Last Name')[planet].
                agg(lambda x: x.mode()[0] if not x.empty else None).to_dict())
    
    for planet in ['HomePlanet_Earth', 'HomePlanet_Mars', 'HomePlanet_Europa']:
        Family_to_HomePlanet = Family_to_planet(planet)
        def fill_HomePlanet(row):
            if pd.isnull(row[planet]):
                return Family_to_HomePlanet.get(row['Last Name'], row[planet])
            return row[planet]
            
        df[planet] = df.apply(fill_HomePlanet, axis = 1)
        
    def Group_to_planet(planet):
        return (df.dropna(subset = [planet]).groupby('GroupId')[planet].
                agg(lambda x: x.mode()[0] if not x.empty else None).to_dict())
    
    for planet in ['HomePlanet_Earth', 'HomePlanet_Mars', 'HomePlanet_Europa']:
        Group_to_HomePlanet = Group_to_planet(planet)
        def fill_HomePlanet(row):
            if pd.isnull(row[planet]):
                return Group_to_HomePlanet.get(row['GroupId'], row[planet])
            return row[planet]
            
        df[planet] = df.apply(fill_HomePlanet, axis = 1)

    deck_Features = [col for col in df.columns if col.startswith('deck_')]
    HomePlanet_Features = [col for col in df.columns if col.startswith('HomePlanet_')]

    for planet in HomePlanet_Features:
        mask = df[planet] == 1
        df.loc[mask, 'HomePlanet'] = df.loc[mask, 'HomePlanet'].fillna(planet.split("_", 1)[1])

    deck_mask = (df[deck_Features].isna().sum(axis = 1) == 1) & ((df[deck_Features] == 0).sum(axis = 1) == len(deck_Features) - 1)
    for d in deck_Features:
        mask = deck_mask & df[d].isnull()
        df.loc[mask, d] = df.loc[mask, d].fillna(1)
        df.loc[mask, 'deck'] = df.loc[mask, 'deck'].fillna(d.split("_", 1)[1])

    df['HomePlanetDeck'] = df.HomePlanet + df.deck
    df = Pre_Prep_Data(df, ['HomePlanetDeck'], [])
    
    return df

def Impute_df(df_orig, df_test_orig, impute_method = 'flag', flag_features = [], columns_to_drop = [], regions_bin_edges = [0, 316, 758, 1137, 1516]):
    df = df_orig.copy()
    df_test = df_test_orig.copy()
    
    df_test['Transported'] = np.nan
    df_all = pd.concat([df, df_test])
    
    if impute_method == 'flag':
        with pd.option_context('future.no_silent_downcasting', True):
            df_all_imputed = df_all.copy().drop(columns = columns_to_drop).fillna(-1).infer_objects(copy = False)
    elif (impute_method == 'Impute') | (impute_method == 'Impute_flag'):
        if impute_method == 'Impute_flag':
            df_all_imputed = Flag_Null(df_all, flag_features)
        else:
            df_all_imputed = df_all.copy()
        df_all_imputed = Impute_Cabins(df_all_imputed, regions_bin_edges)
        df_all_imputed = Impute_Family(df_all_imputed)
        df_all_imputed = Impute_Planets(df_all_imputed)
        df_all_imputed = Impute_CryoSleep_VIP(df_all_imputed)
        df_all_imputed = Impute_Age_Luxury(df_all_imputed)
        df_all_imputed.drop(columns = columns_to_drop, inplace = True)

        recalc_cols = ['CabinFamilySize', 'CabinGroupSize', 'CabinSize', 'FirstNameLength', 'LastNameLength', 'GroupFamilySize', 'FamilySize']
        skewness = df_all_imputed[recalc_cols].skew().apply(np.abs)
        skewed_features = skewness[skewness > 0.6].index.to_list()
        df_all_imputed[skewed_features] = df_all_imputed[skewed_features].apply(np.log1p)
    
    eps = 1e-16
    int_features = list({f for f in df_all.select_dtypes(include = 'number').columns if df_all[f].dropna().mod(1).abs().lt(eps).all()} - set(columns_to_drop))
    bool_features = list({f for f in df_all.select_dtypes(include = 'object').columns if set(df_all[f].dropna().unique()).issubset({'True', 'False', 0, 1})} - set(columns_to_drop) - {'Transported'})
    df_all_imputed[int_features] = df_all_imputed[int_features].round(0).astype(int)
    df_all_imputed[bool_features] = df_all_imputed[bool_features].astype(int)
    
    df = df_all_imputed.iloc[: len(df)].copy(deep = True)
    df.Transported = df.Transported.astype(int)
    df_test = (df_all_imputed.iloc[-len(df_test) :]).drop(columns = ['Transported']).copy(deep = True)
    
    return df, df_test

def Flag_Null(df_orig, flag_features):
    df = df_orig.copy()
    for feature in flag_features:
        if feature == 'Last Name':
            flag_feature = 'flag_Name'
        else:
            flag_feature = 'flag_' + feature
            
        df[flag_feature] = np.where(df[feature].isnull(), 1, 0)
    
    return df

def Impute_Cabins(df_orig, regions_bin_edges):
    df = df_orig.copy()
    deck_Features = ['deck_A', 'deck_B', 'deck_C', 'deck_T', 'deck_G', 'deck_F', 'deck_D', 'deck_E']
    side_Features = ['side_S', 'side_P']
    
    for nround in range(1, 7):
        for deck in deck_Features:
            if deck in ['deck_A', 'deck_B', 'deck_C', 'deck_T']:
                planets_to_mask = {'HomePlanet_Europa': 1}
            elif deck == 'deck_G':
                planets_to_mask = {'HomePlanet_Earth': 1}
            elif deck == 'deck_D':
                planets_to_mask = {'HomePlanet_Earth': 0}
            elif deck == 'deck_F':
                planets_to_mask = {'HomePlanet_Europa': 0}
            elif deck == 'deck_E':
                planets_to_mask = {}
            for side in side_Features:
                df = FindFill_missing_cabins(df, deck, side, planets_to_mask, nround)
                
    missing_group_cabins = df[df['Cabin'].isnull()]['GroupId'].unique().tolist()

    mask = df['GroupId'].isin(missing_group_cabins)

    Group_to_Cabin = (df[mask].dropna(subset = ['Cabin']).groupby('GroupId')['Cabin'].agg(
        lambda x: x.mode().iloc[-1] if not x.empty else None).to_dict())

    def fill_Cabin(row):
        if pd.isnull(row['Cabin']):
            return Group_to_Cabin.get(row['GroupId'], row['Cabin'])
        return row['Cabin']
        
    df['Cabin'] = df.apply(fill_Cabin, axis = 1)
    df_split = df.Cabin.str.split("/", expand = True).rename({0: 'deck', 1: 'Cabin Number', 2: 'side'}, axis = 1)
    df['deck'] = df_split['deck'].values
    df['side'] = df_split['side'].values
    df['Cabin Number'] = df_split['Cabin Number'].values.astype(int)
    
    deck_Features = [col for col in df.columns if col.startswith('deck_')]
    side_Features = [col for col in df.columns if col.startswith('side_')]
    
    df[deck_Features] = pd.get_dummies(df['deck'], prefix = 'deck').astype(int)
    df[side_Features] = pd.get_dummies(df['side'], prefix = 'side').astype(int)
    df = Get_Regions(df, 'Cabin Number', 'Region', regions_bin_edges)
    
    df['CabinFamilySize'] = df.groupby(['Last Name', 'Cabin'])['Cabin'].transform(len)
    df['CabinGroupSize'] = df.groupby(['GroupId', 'Cabin'])['Cabin'].transform(len)
    df['CabinSize'] = df.groupby('Cabin')['Cabin'].transform(len)
    
    mask = df['CabinSize'] == 1
    df.loc[mask, 'CabinFamilySize'] = df.loc[mask, 'CabinFamilySize'].fillna(1)
    
    return df

def FindFill_missing_cabins(df_orig, deck, side, planets_to_mask, nround):
    deck_Features = ['deck_A', 'deck_B', 'deck_C', 'deck_D', 'deck_E', 'deck_F', 'deck_G', 'deck_T']
    side_Features = ['side_S', 'side_P']
    df = df_orig.copy()
    mask = (df[deck] == 1) & (df[side] == 1)
    cabins_list = sorted(df[mask]['Cabin Number'].unique())
    range_cabins = set(range(min(cabins_list), max(cabins_list) + 1))
    cabins_set = set(cabins_list)
    missing_cabins = sorted(range_cabins - cabins_set)
    
    def get_groups(cabin):
        if cabin - 1 not in missing_cabins:
            prev_cabin = cabin - 1
        else:
            prev_cabin = cabin - 2
        if cabin + 1 not in missing_cabins:
            next_cabin = cabin + 1
        else:
            next_cabin = cabin + 2
        
        group_min = df[mask & (df['Cabin Number'] == prev_cabin)]['GroupId'].values[0]
        group_max = df[mask & (df['Cabin Number'] == next_cabin)]['GroupId'].values[0]
        
        return group_min, group_max
    
    def get_mask(group_min, group_max, cond):
        group_mask = (df['GroupId'] >= group_min) & (df['GroupId'] <= group_max) & (df['Cabin Number'].isnull())
        group_g1 = df['GroupSize'] == np.log1p(1)
        if not planets_to_mask:
            group_planet_mask = group_mask
        elif len(planets_to_mask) == 1:
            for key in planets_to_mask.keys():
                planet_mask = df[key] == planets_to_mask[key]
            group_planet_mask = group_mask & planet_mask
        else:
            print('error: planets_to_mask dictionary should not have more than one entry')
            return 0
        if cond == 1:
            return group_planet_mask
        elif cond == 2:
            return group_planet_mask & group_g1
        
    def fill_cabin(cabin, index_to_fill, cond):
        df.at[index_to_fill, 'Cabin Number'] = cabin
        df.at[index_to_fill, 'deck'] = deck[-1]
        df.at[index_to_fill, 'side'] = side[-1]
        cabin_name = deck[-1] + '/' + str(cabin) + '/' + side[-1]
        df.at[index_to_fill, 'Cabin'] = cabin_name
        for d in deck_Features:
            if d == deck:
                df.at[index_to_fill, d] = 1
            else:
                df.at[index_to_fill, d] = 0
        for s in side_Features:
            if s == side:
                df.at[index_to_fill, s] = 1
            else:
                df.at[index_to_fill, s] = 0
                
#         print(f'cond{cond}: Passenger {index_to_fill} placed in cabin {cabin_name}')
    
    for cabin in missing_cabins:
        group_min, group_max = get_groups(cabin)
        cond1_mask = get_mask(group_min, group_max, 1)
        null_cabin_rows = df[cond1_mask]
        cond2_mask = get_mask(group_min, group_max, 2)
        null_cabin_rows_g1 = df[cond2_mask]
        
        condition1 = len(null_cabin_rows) == 1
        condition2 = (nround > 2) & (len(null_cabin_rows_g1) == 1)
        condition3 = (nround > 4) & (len(null_cabin_rows_g1) > 1)
        
        if condition1:
            fill_cabin(cabin, null_cabin_rows.index[0], 1)
        elif condition2:
            fill_cabin(cabin, null_cabin_rows_g1.index[0], 2)
        elif condition3:
            fill_cabin(cabin, null_cabin_rows_g1.index[0], 3)
        elif nround > 5:
            groups_in_cabin = set([null_cabin_rows.index[k][:4] for k in range(len(null_cabin_rows))])
            if len(groups_in_cabin) == 1:
                for k in range(len(null_cabin_rows)):
                    fill_cabin(cabin, null_cabin_rows.index[k], 4)
    return df

def Impute_Family(df_orig):
    df = df_orig.copy()
    
    # For passengers sharing a cabin, set Last Name, as equal to that of the family with the largest members in that cabin. 
    # FirstNameLength = mod in the cabin also
    
    cabin_mask = df['CabinSize'] > 1
    null_mask = df['Last Name'].isnull()
    missing_cabin_names = df[cabin_mask & null_mask]['Cabin'].unique().tolist()

    mask = df['Cabin'].isin(missing_cabin_names)

    def fill_LastName(row):
        if (pd.isnull(row['Last Name'])) & (row['CabinSize'] > 1):
            return df[df['Cabin'] == row['Cabin']]['Last Name'].mode()[0]
        return row['Last Name']

    df['Last Name'] = df.apply(fill_LastName, axis = 1)

    def fill_FirstName(row):
        if (pd.isnull(row['FirstNameLength'])) & (row['CabinSize'] > 1):
            return df[df['Cabin'] == row['Cabin']]['FirstNameLength'].mode()[0]
        return row['FirstNameLength']

    df['FirstNameLength'] = df.apply(fill_FirstName, axis = 1)
    
    df['LastNameLength'] = df['Last Name'].fillna('').astype(str).apply(len).replace(0, np.nan)
    df['GroupFamilySize'] = df.groupby(['Last Name', 'GroupId'])['Last Name'].transform(len)
    df['CabinFamilySize'] = df.groupby(['Last Name', 'Cabin'])['Cabin'].transform(len)
    df['FamilySize'] = df.groupby('Last Name')['Last Name'].transform(len)

    fmask = df['FamilySize'] == 1
    df.loc[fmask, 'CabinFamilySize'] = df.loc[fmask, 'CabinFamilySize'].fillna(1)
    df.loc[fmask, 'GroupFamilySize'] = df.loc[fmask, 'GroupFamilySize'].fillna(1)

    cmask = df['CabinSize'] == 1
    df.loc[cmask, 'CabinFamilySize'] = df.loc[cmask, 'CabinFamilySize'].fillna(1)
    
    gmask = df['GroupSize'] == np.log1p(1)
    df.loc[gmask, 'GroupFamilySize'] = df.loc[gmask, 'GroupFamilySize'].fillna(1)
    
    # For the rest of the passengers, fillna with K-nearest neighbors imputations
    Destination_Features = [col for col in df.columns if col.startswith('Destination_')]
    HomePlanet_Features = [col for col in df.columns if col.startswith('HomePlanet_')]
    deck_Features = [col for col in df.columns if col.startswith('deck_')]
    Luxury_Features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    Imp_Features = ['FamilySize', 'FirstNameLength', 'LastNameLength', 'GroupFamilySize']
    Dep_Features = (['GroupSize', 'Age'] + 
                    Destination_Features + 
                    HomePlanet_Features + 
                    deck_Features + 
                    Luxury_Features)
    High_Dep_Features = HomePlanet_Features + ['GroupSize']
    
    df = KNN_Impute_Features(df, Imp_Features, Dep_Features, High_Dep_Features)
    
    df[Imp_Features] = df[Imp_Features].round().astype(int)
    
    return df

def Impute_Planets(df_orig):
    df = df_orig.copy()
    
    # Impute via knn
    Destination_Features = [col for col in df.columns if col.startswith('Destination_')]
    HomePlanet_Features = [col for col in df.columns if col.startswith('HomePlanet_')]
    deck_Features = [col for col in df.columns if col.startswith('deck_')]
    HomePlanetDeck_Features = [col for col in df.columns if col.startswith('HomePlanetDeck_')] 
    Luxury_Features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    Imp_Features = HomePlanet_Features + Destination_Features
    Dep_Features = (['CryoSleep', 'GroupSize', 'Age', 'CabinSize', 'CabinFamilySize', 'FamilySize'] + 
                    deck_Features + Luxury_Features)
    High_Dep_Features = deck_Features + HomePlanet_Features
    
    df = KNN_Impute_Features(df, Imp_Features, Dep_Features, High_Dep_Features, n_neighbors = 9)
    
    # First, let's attempt to break any ties. 
    # add a very small random number to Destination TRAPPIST, r = [esp, 1.5eps]
    # Next add a smaller one to Destination Cancri if HomePlanet is Mars or Europa
    # or to Destination PSO if HomePlanet is Earth
    
    eps = 0.001
    df['Destination_TRAPPIST-1e'] += np.random.uniform(eps, 1.5*eps, size = len(df))
    df['eps'] = np.where(df['HomePlanet_Earth'] == 1, np.random.uniform(0.5*eps, eps), 0)
    df['Destination_PSO J318.5-22'] += df['eps']
    df['eps'] = np.where(df['HomePlanet_Earth'] == 0, np.random.uniform(0.5*eps, eps), 0)
    df['Destination_55 Cancri e'] += df['eps']
    df.drop(columns = ['eps'], inplace = True)
    
    # Now, let's normalize HomePlanet and Destination Features
    df[HomePlanet_Features] = df[HomePlanet_Features].div(df[HomePlanet_Features].sum(axis = 1), axis = 0)
    df[Destination_Features] = df[Destination_Features].div(df[Destination_Features].sum(axis = 1), axis = 0)
        
    # Next, pick planet with highest probability and set the rest to false
    df[HomePlanet_Features] = df[HomePlanet_Features].eq(df[HomePlanet_Features].max(axis = 1), axis = 0).astype(int)
    df[Destination_Features] = df[Destination_Features].eq(df[Destination_Features].max(axis = 1), axis = 0).astype(int)
    
    # Fill in HomePlanet and Destination (even though these will be dropped later ...)
    for planet in HomePlanet_Features:
        mask = df[planet] == 1
        df.loc[mask, 'HomePlanet'] = df.loc[mask, 'HomePlanet'].fillna(planet.split('HomePlanet_')[1])
        
    for planet in Destination_Features:
        mask = df[planet] == 1
        df.loc[mask, 'Destination'] = df.loc[mask, 'Destination'].fillna(planet.split('Destination_')[1])

    # Assuming deck is already imputed, we can now updare HomePlanetDeck Features
    df.HomePlanetDeck = df.HomePlanet + df.deck
    mask = df[HomePlanetDeck_Features].isnull().any(axis = 1)
    for HD in HomePlanetDeck_Features:
        df[HD] = df[HD].fillna((df.HomePlanetDeck == HD.split("_", 1)[1]).astype(int))
        
    return df

def Impute_CryoSleep_VIP(df_orig):
    df = df_orig.copy()
    
    mask_age = df['Age'] > 12
    df = fill_NA(df, mask_age, 'CryoSleep', 1)
    # df.loc[mask_age, 'CryoSleep'] = df.loc[mask_age, 'CryoSleep'].fillna(1)
    
    mask_dest = (df['Destination_55 Cancri e'] == 1) | (df['Destination_PSO J318.5-22'] == 1)
    mask_age_child = df['Age'] <= 12
    # df.loc[mask_dest & mask_age_child, 'CryoSleep'] = df.loc[mask_dest & mask_age_child, 'CryoSleep'].fillna(0)
    df = fill_NA(df, mask_dest & mask_age_child, 'CryoSleep', 0)
    
    mask_dest = df['Destination_TRAPPIST-1e'] == 1
    
    mask_home = df['HomePlanet_Europa'] == 1
    mask = mask_dest & mask_age_child & mask_home
    # df.loc[mask, 'CryoSleep'] = df.loc[mask, 'CryoSleep'].fillna(0)
    df = fill_NA(df, mask, 'CryoSleep', 0)
    
    mask_home = df['HomePlanet_Mars'] == 1
    mask = mask_dest & mask_home & (df['Age'] > 5)
    # df.loc[mask, 'CryoSleep'] = df.loc[mask, 'CryoSleep'].fillna(1)
    df = fill_NA(df, mask, 'CryoSleep', 1)
    
    mask_home = df['HomePlanet_Earth'] == 1
    mask = mask_dest & mask_home & (df['Age'] > 7) & (df['Age'] < 12)
    # df.loc[mask, 'CryoSleep'] = df.loc[mask, 'CryoSleep'].fillna(0)
    df = fill_NA(df, mask, 'CryoSleep', 0)
    
    mask = mask_home & mask_dest & (df['Age'] == 0) & (df['GroupFamilySize'] == 6)
    # df.loc[mask, 'CryoSleep'] = df.loc[mask, 'CryoSleep'].fillna(1)
    df = fill_NA(df, mask, 'CryoSleep', 1)
    
    mask = mask_home & mask_dest & (df['Age'] == 3)
    # df.loc[mask, 'CryoSleep'] = df.loc[mask, 'CryoSleep'].fillna(1)
    df = fill_NA(df, mask, 'CryoSleep', 1)
    
    mask = mask_home & mask_dest & (df['Age'] == 1) & (df['CabinFamilySize'] == 4)
    # df.loc[mask, 'CryoSleep'] = df.loc[mask, 'CryoSleep'].fillna(1)
    df = fill_NA(df, mask, 'CryoSleep', 1)
    
    mask = mask_home & mask_dest & (df['Age'] == 2) & (df['CabinFamilySize'] == 3)
    # df.loc[mask, 'CryoSleep'] = df.loc[mask, 'CryoSleep'].fillna(1)
    df = fill_NA(df, mask, 'CryoSleep', 1)
    
    # For the rest of the passengers, fill na CryoSleep with K-nearest neighbors imputations
    # Fill all na VIP with K-nearest neighbors imputations
    
    Destination_Features = [col for col in df.columns if col.startswith('Destination_')]
    HomePlanet_Features = [col for col in df.columns if col.startswith('HomePlanet_')]
    deck_Features = [col for col in df.columns if col.startswith('deck_')]
    Luxury_Features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    Imp_Features = ['CryoSleep', 'VIP']
    Dep_Features = (['GroupSize', 'Age', 'CabinSize', 'CabinFamilySize'] + 
                    Destination_Features + 
                    HomePlanet_Features + 
                    deck_Features + 
                    Luxury_Features)
    High_Dep_Features = Luxury_Features + ['Age']
    
    df = KNN_Impute_Features(df, Imp_Features, Dep_Features, High_Dep_Features)
    
    df[Imp_Features] = df[Imp_Features].round().astype(int)
    
    return df

def Impute_Age_Luxury(df_orig):
    df = df_orig.copy()
    
    # Impute via knn
    Destination_Features = [col for col in df.columns if col.startswith('Destination_')]
    HomePlanet_Features = [col for col in df.columns if col.startswith('HomePlanet_')]
    deck_Features = [col for col in df.columns if col.startswith('deck_')]
    Luxury_Features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    Imp_Features = Luxury_Features + ['Age']
    Dep_Features = (deck_Features + HomePlanet_Features + Destination_Features + ['CryoSleep', 'GroupSize'])
    High_Dep_Features = Luxury_Features + ['Age', 'CryoSleep']
    
    df = KNN_Impute_Features(df, Imp_Features, Dep_Features, High_Dep_Features)

    df.Under_13 = df.Age < 13

    bin_edges = [0, 2, 5, 13, 20, 36, 46, 65, 90]
    LifeStages_choices = ['Infant', 'Toddler', 'Child', 'Teen', 'Young Adult', 'Young middle aged', 'Older middle aged', 'Senior']
    df['Age_LifeStages_lbs'] = pd.cut(df.Age, bins = bin_edges, labels = LifeStages_choices, right = False, include_lowest = True)
    df['Age_LifeStages'] = (df.Age_LifeStages_lbs.map({val: idx for idx, val in enumerate(LifeStages_choices)})).astype(int)
    
    return df

def KNN_Impute_Features(df_orig, Imp_Features, Dep_Features, High_Dep_Features, n_neighbors = 5):
    df = df_orig.copy()
    
    imputer = KNNImputer(n_neighbors = n_neighbors, weights = 'distance')
    df_imputed = df[Imp_Features + Dep_Features].copy()
    
    scaler = RobustScaler()
    scaler.fit(df_imputed)
    df_imputed = pd.DataFrame(scaler.transform(df_imputed), columns = df_imputed.columns)
    
    df_imputed[High_Dep_Features] *= 2
    df_imputed = pd.DataFrame(imputer.fit_transform(df_imputed), columns = df_imputed.columns)
    df_imputed[High_Dep_Features] /= 2
    
    df_imputed = pd.DataFrame(scaler.inverse_transform(df_imputed), columns = df_imputed.columns)
    
    df[Imp_Features] = df_imputed[Imp_Features].values
    
    return df

class FeatureImportance:
    def __init__(self, X, y, Binary_Categorical = [], Ordinal_Categorical = [], Nominal_Categorical = [], random_state = 3):
        self.X = X.copy()
        self.y = y.copy()
        self.random_state = random_state
        self.Cat_Features = Binary_Categorical + Ordinal_Categorical + Nominal_Categorical
        self.Nom_Cat_Features = Nominal_Categorical
        assert(set(self.Cat_Features).issubset(set(self.X.columns)))
        self.ohe_cols = [f'{col}_{c}' for col in self.Nom_Cat_Features for c in self.X[col].unique()]
        self.catboost_params = {'iterations': 1000, 'learning_rate': 0.02, 'od_wait': 100, 'eval_metric': 'Accuracy', 
                                'bootstrap_type': 'Bayesian', 'loss_function': 'Logloss', 'auto_class_weights': 'Balanced'}

    def _mi_input(self):
        cols_to_drop = list(set(self.ohe_cols) & set(self.X.columns))
        X_clean = self.X.drop(columns = cols_to_drop).copy()
        enc = OrdinalEncoder()
        X_clean[self.Nom_Cat_Features] = enc.fit_transform(X_clean[self.Nom_Cat_Features])
        discrete_mask = [feature in self.Cat_Features for feature in X_clean.columns]
        return X_clean, discrete_mask

    def _lasso_input(self):
        if set(self.ohe_cols).issubset(set(self.X.columns)):
            X_clean = self.X.drop(columns = self.Nom_Cat_Features).copy()
        else:
            cols_to_drop = list(set(self.ohe_cols) & set(self.X.columns))
            X_clean = self.X.drop(columns = cols_to_drop).copy()
            for feature in self.Nom_Cat_Features:
                one_hot_encoded = pd.get_dummies(X_clean[feature], prefix = feature)
                X_clean = pd.concat([X_clean, one_hot_encoded], axis = 1).drop(columns = self.Nom_Cat_Features)
        return X_clean

    def mutual_info(self, random_state = None):
        rs = self.random_state if random_state is None else random_state
        X_mi, discrete_mask = self._mi_input()
        MI = mutual_info_classif(X_mi, self.y, discrete_features = discrete_mask, random_state = rs)
        return pd.Series(MI, index = X_mi.columns.tolist()).sort_values(ascending = False)

    def LCV_importance(self):
        X_lcv = self._lasso_input()
        lasso_cv = LassoCV(cv = 5)
        lasso_cv.fit(X_lcv, self.y)
        importance = np.abs(lasso_cv.coef_)
        return pd.Series(importance, index = X_lcv.columns.tolist()).sort_values(ascending = False)

    def catboost_perm_importance_cv(self, n_splits = 5, n_repeats = 3, n_repeats_pi = 100, model_params = None, random_state = None):
        rs = self.random_state if random_state is None else random_state
        X_clean, _ = self._mi_input()
        params = self.catboost_params if model_params is None else model_params
        
        cv = RepeatedStratifiedKFold(n_splits = n_splits, n_repeats = n_repeats, random_state = rs)
        features_importances = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_clean, self.y)):
            X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            
            model = CatBoostClassifier(**params, verbose = False)
            model.fit(X_train, y_train, eval_set = (X_val, y_val), use_best_model = True, verbose = False)

            r = permutation_importance(model, X_val, y_val, n_repeats = n_repeats_pi)
            features_importances.append(r['importances_mean'])
        
        return pd.Series(np.vstack(features_importances).mean(axis = 0), index = X_clean.columns.tolist()).sort_values(ascending = False)

    def catboost_rfecv(self, step = 1, cv = 5, model_params = None, random_state = None):
        rs = self.random_state if random_state is None else random_state
        X_clean, _ = self._mi_input()
        params = self.catboost_params if model_params is None else model_params
        model = CatBoostClassifier(**params, verbose = False)
        selector = RFECV(model, step = step, cv = cv)
        selector = selector.fit(X_clean, self.y)
        Features = np.array(X_clean.columns.to_list())
        Selected_Features = Features[selector.support_]
        return Selected_Features

    def catboost_emb(self, model_params = None):
        X_clean, _ = self._mi_input()
        params = self.catboost_params if model_params is None else model_params
        model = CatBoostClassifier(**params, verbose = False)
        model.fit(X_clean, self.y)
        return model.get_feature_importance(type = "PredictionValuesChange", prettified = True).set_index('Feature Id')

    def Get_PCA_Input(self, v = 0.9):
        scaler = RobustScaler()
        X_clean = self._lasso_input()
        X_scaled = scaler.fit_transform(X_clean)
        if v == 1.0:
            return X_scaled
        else:
            pca = PCA(n_components = v)
            return pca.fit_transform(X_scaled)