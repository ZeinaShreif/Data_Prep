import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import numpy as np
import string

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

def Extract_Initial_Data(df_orig, df_test_orig, version = 1, regions_bin_edges = [0, 316, 758, 1137, 1516]):
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
        bin_edges = [1, 3713, 4641, 6497, 7425]
        df_all = Get_Regions(df_all, 'GroupId', 'Batch', bin_edges)
        df_all = Get_Regions(df_all, 'Cabin Number', 'Region', regions_bin_edges)
        df_all.drop(columns = list(string.ascii_lowercase) + 
                    ['G1', 'G2', 'G3', 'G4', 'Name', 'First Name', 'CabinGroupSize'], 
                    inplace = True)
        
    df = df_all.iloc[: len(df)]
    df_test = df_all.iloc[-len(df_test) :]
    return df, df_test.drop(columns = ['Transported'])

def fill_NA(df, mask, col, value):
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
    df.loc[mask, 'CryoSleep'] = df.loc[mask, 'CryoSleep'].fillna(False)
    
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