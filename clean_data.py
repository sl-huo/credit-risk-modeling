
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)

def summarytable(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values
    summary['Missing_Percentage'] = np.round(df.isnull().sum().values/df.shape[0],2)
    summary['Uniques'] = df.nunique().values
    summary['Uniques_Percentage'] = np.round(df.nunique().values/df.shape[0],2)
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values
    summary['Last Value'] = df.iloc[-1].values

    return summary


# comparing distribution of categorical/object variables between two target classes
def category_plot(df, col_name, targ_var):
    count_df = df.groupby([str(col_name), str(targ_var)])[str(col_name)].count().unstack()
    distribution_df = pd.crosstab(df[str(col_name)], df[str(targ_var)], normalize='index')
    my_colors=['cornflowerblue', 'lightsalmon']
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (15,4))
    count_df.plot(kind='barh', stacked=True, ax=axes[0], color=my_colors, title=f'Number of loans - {col_name}')
    distribution_df.plot(kind='barh', stacked=True, ax=axes[1], color=my_colors, title=f'Percentage of default loans - {col_name}')
    plt.legend(labels=['Not-default', 'Default'])
    plt.tight_layout()
    plt.show()

# plotting histogram for numeric variables with kde
def hist_plot(df, col_name):
    sns.histplot(data=df[col_name], color='cornflowerblue', kde=True, edgecolor='white')
    plt.axvline(df[col_name].mean(), c='orange', label='Mean')
    plt.title(f'Distribution of {col_name}')
    plt.ylabel('# of loans')
    plt.legend()
    
# plotting for batches
def histcompare_plot(df, col_name, targ_var):
    sns.histplot(data=df[df[targ_var]==0], x=col_name, label='Not-default', bins=25, stat='probability', color='cornflowerblue', edgecolor='white', alpha=0.5)
    sns.histplot(data=df[df[targ_var]==1], x=col_name, label='Default', bins=25, stat='probability', color='lightsalmon', edgecolor='white', alpha=0.5)
    plt.title(f'Distribution of {col_name}')
    plt.legend()


def af_data_clean(df, id_vars, targ_var):
    # reference: https://github.com/rapidsai-community/showcase/tree/main/event_notebooks/GTC_2021/credit_scorecard/cpu
    ## Sort data by unique client identifier
    df.sort_values(id_vars, inplace=True)
    df.reset_index(drop=True, inplace=True)

    ## Make the target variable the second column in the dataframe
    targets = df.pop(targ_var)
    df.insert(1, targ_var, targets)

    ## Replace periods in variable names with underscores 
    new_cols = [sub.replace('.', '_') for sub in df.columns] 
    df.rename( columns=dict(zip(df.columns, new_cols)), inplace=True)

    ## Specify variables that should be treated as categorical and convert them to character strings (non-numeric)
    cat_vars = [ 'branch_id', 'supplier_id', 'manufacturer_id', 'Current_pincode_ID', 'State_ID', 'Employee_code_ID'
                , 'MobileNo_Avl_Flag', 'Aadhar_flag', 'PAN_flag', 'VoterID_flag', 'Driving_flag', 'Passport_flag']
    df[cat_vars] = df[cat_vars].fillna('')
    df[cat_vars] = df[cat_vars].applymap(str)

    ## Strategically add some missing data 
    ## Note: There is no bureau data for more than half of the records
    no_bureau = (df.PERFORM_CNS_SCORE_DESCRIPTION == 'No Bureau History Available')
    df.loc[no_bureau, 'PERFORM_CNS_SCORE_DESCRIPTION'] = ''
    bureau_vars = [ 'PERFORM_CNS_SCORE', 'PRI_NO_OF_ACCTS', 'PRI_ACTIVE_ACCTS', 'PRI_OVERDUE_ACCTS'
                   , 'PRI_CURRENT_BALANCE', 'PRI_SANCTIONED_AMOUNT', 'PRI_DISBURSED_AMOUNT', 'PRIMARY_INSTAL_AMT']
    df.loc[no_bureau, bureau_vars] = np.nan

    ## The 'Credit Score' variable PERFORM_CNS_SCORE has some issues and could use some additional feature engineering.
    ## The values of 300, 738, and 825 are over-represented in the data (300 should be at the end of the distribution)
    ## The values 11,14-18 are clearly 'Not Scored' codes - setting to missing for demo
    # df.PERFORM_CNS_SCORE.value_counts()
    # df.PERFORM_CNS_SCORE_DESCRIPTION.value_counts().sort_index()
    # pd.crosstab(df.PERFORM_CNS_SCORE_DESCRIPTION, df.PERFORM_CNS_SCORE, margins=True)
    df.loc[df.PERFORM_CNS_SCORE < 20, 'PERFORM_CNS_SCORE'] = np.nan

    ## Make all date calculation relative to January 2019 when this dataset was created.
    today = pd.to_datetime('201901', format='%Y%m')
    df['DoB'] = pd.to_datetime(df['Date_of_Birth'], format='%d-%m-%y', errors='coerce')
    # convert year after 2019 to before 2019
    df['DoB'] = df['DoB'].mask( df['DoB'].dt.year > today.year
                                        , df['DoB'] - pd.offsets.DateOffset(years=100))
    # convert BoD to age in year
    df['AgeInYear'] = (today - df.DoB).astype('timedelta64[Y]')
    # convert disbursement date to days since disbursement
    df['DaysSinceDisbursement'] = (today - pd.to_datetime(df.DisbursalDate, format='%d-%m-%y')
                                       ).astype('timedelta64[D]')

    def timestr_to_mths(timestr):
        '''timestr formatted as 'Xyrs Ymon' '''
        year = int(timestr.split()[0].split('y')[0]) 
        mo = int(timestr.split()[1].split('m')[0])
        num = year*12 + mo
        return(num)
    
    # convert object timestring to int months
    df['AcctAgeInMonths'] = df['AVERAGE_ACCT_AGE'].apply(lambda x: timestr_to_mths(x))
    df['CreditHistLenInMonths'] = df["CREDIT_HISTORY_LENGTH"].apply(lambda x: timestr_to_mths(x))

    ## Drop some variables that have been converted
    # drop 'MobileNo_Avl_Flag' because it is all 1
    df = df.drop(columns=['Date_of_Birth', 'DoB', 'AVERAGE_ACCT_AGE', 'CREDIT_HISTORY_LENGTH', 'MobileNo_Avl_Flag', 'DisbursalDate'] )
    df[targ_var] = df[targ_var].astype(int)

    # ## Can drop records with no credit history - just to trim the data (justifiable in scenarios where
    # ## no_credit_bureau leads to an auto-decline or initiates a separate adjudication process)
    # df = df.loc[(~no_bureau | (df.SEC_NO_OF_ACCTS != 0)), :]
    
    ## Drop some variables that are not good for scorecarding (sparse, high cardinality)
    ## check the value counts for the top 10 categories for sparse variables
    # sparse_cat_vars = ['Current_pincode_ID', 'Employee_code_ID','branch_id','manufacturer_id','State_ID', 'supplier_id']
    # for i in sparse_cat_vars:
    #   print(df_loan_clean[i].value_counts(normalize=True).head(10))
    ## The variable 'branch_id' is likely linked to geography and therefore demographics
    df = df.drop(columns=['supplier_id', 'Current_pincode_ID', 'Employee_code_ID', 'branch_id'])
    
    # ## Give some variables shorter names 
    # df.rename(columns={'PERFORM_CNS_SCORE_DESCRIPTION': 'PERF_CNS_SC_DESC'
    #                , 'DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS': 'DELI_ACCTS_LAST_6_MTHS'
    #                , 'NEW_ACCTS_IN_LAST_SIX_MONTHS': 'NEW_ACCTS_LAST_6_MTHS'}, inplace=True)
    
    return df


