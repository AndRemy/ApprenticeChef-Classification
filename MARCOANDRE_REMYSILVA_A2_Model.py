# timeit

# Student Name : Marco Andre Remy Silva
# Cohort       : Haight - 2

# Note: You are only allowed to submit ONE final model for this assignment.


################################################################################
# Import Packages
################################################################################

# use this space for all of your package imports
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import seaborn           as sns

from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import roc_auc_score, roc_curve, confusion_matrix, classification_report




################################################################################
# Load Data
################################################################################

# use this space to load the original dataset
# MAKE SURE TO SAVE THE ORIGINAL FILE AS original_df
# Example: original_df = pd.read_excel('Apprentice Chef Dataset.xlsx')

original_df = pd.read_excel("Apprentice_Chef_Dataset.xlsx")


################################################################################
# Feature Engineering and (optional) Dataset Standardization
################################################################################

# use this space for all of the feature engineering that is required for your
# final model

# if your final model requires dataset standardization, do this here as well

TEST_SIZE   = 0.25
SEED        = 222

def Model_Execution(x_label = None, model = LogisticRegression(), scaled = False):
    """
    The function splits the data and runs the model sent as parameter.
    
    PARAMETERS
    ---------------------
    x_label = All the dependent variables
    model   = The model to run. (Default: Logistic Regression)
    scaled  = Determines if the data will be executed or not. (Default: False)
    """
    global TEST_SIZE
    global SEED
    global original_df
    
    y_label = ["CROSS_SELL_SUCCESS"]

    X_data = original_df.loc[:, x_label]
    y_data = original_df.loc[:, y_label]
    
    if(scaled):
        #Standardizing the data
        scaler = StandardScaler()
        scaler.fit(X_data)
        X_scaled = scaler.transform(X_data)
        X_data   = X_scaled

    X_train, X_test, y_train, y_test = train_test_split(X_data,
                                                        y_data,
                                                        test_size    = TEST_SIZE,
                                                        random_state = SEED,
                                                        stratify     = y_data)
    if(scaled):
        print("""
    ##############################################################################
    ###                         SCALED REGRESSION                              ###
    ##############################################################################
        """)
    else:
        print("""
    ##############################################################################
    ###                        UNSCALED REGRESSION                             ###
    ##############################################################################
        """)
    
    return retrieve_results(X_train, X_test, y_train, y_test, model)

##############################################################################

def retrieve_results(X_train = None, X_test = None, y_train = None, y_test = None, model = None):
    """
    It fits, predicts and scores the model.
    
    PARAMETERS
    ---------------------
    X_train: Independent variables to train the model
    X_test : Independent variables to test the model
    y_train: Dependent variable to train the model
    y_test : Dependent variable to test the model
    model  : The model to run
    """
    model.fit(X_train, y_train)
        
    # Compute predicted probabilities: y_pred_prob
    pred_y = model.predict(X_test)

    #Train Score
    train_score = model.score(X_train, y_train).round(3)
    
    #Test Score
    test_score  = model.score(X_test, y_test).round(3)    
    
    #AUC Score
    auc         = roc_auc_score(y_true  = y_test,
                                y_score = pred_y).round(3)
    
    return train_score, test_score, auc

##############################################################################

#Categories of email according to Marketing
email_type = {
    "professional" : [
        '@mmm.com',                 '@amex.com',                '@apple.com',
        '@boeing.com',              '@caterpillar.com',         '@chevron.com',
        '@cisco.com',               '@cocacola.com',            '@disney.com',
        '@dupont.com',              '@exxon.com',               '@ge.org',
        '@goldmansacs.com',         '@homedepot.com',           '@ibm.com',
        '@intel.com',               '@jnj.com',                 '@jpmorgan.com',
        '@mcdonalds.com',           '@merck.com',               '@microsoft.com',
        '@nike.com',                '@pfizer.com',              '@pg.com',
        '@travelers.com',           '@unitedtech.com',          '@unitedhealth.com'
        '@verizon.com',             '@visa.com',                '@walmart.com'
    ],
    "personal" : [
        '@gmail.com',               '@yahoo.com',               '@protonmail.com'
    ],
    "junk" : [
        '@me.com',                  '@aol.com',                 '@hotmail.com',
        '@live.com',                '@msn.com',                 '@passport.com'
    ]
}

def Is_Valid_Email(data):
    """
    Is_Valid_Email(data): Function that determines if a email is considered or not a valie email
    (
        0: Invalid Email
        1: Valid Email
    )
    
    PARAMETERS
    ---------------------
    data: The value of the email.
    """
    result = 1
    data = "@" + data.split("@")[1]
    for key in email_type.keys():
        if data in email_type[key]:
            if (key == 'junk'):
                result = 0
            break
    return result


#Defining if a row has a valid email or not
original_df['IS_VALID_EMAIL'] = original_df["EMAIL"].apply(Is_Valid_Email)

#Setting the trend change for FOLLOWED_RECOMMENDATIONS_PCT. WORKED!!!
change_FOLLOWED_RECOMMENDATIONS_PCT_low = 5
change_FOLLOWED_RECOMMENDATIONS_PCT_hi  = 35
original_df['change_FOLLOWED_RECOMMENDATIONS_PCT'] = 0
condition = original_df.loc[:,'change_FOLLOWED_RECOMMENDATIONS_PCT'][np.logical_and(original_df['FOLLOWED_RECOMMENDATIONS_PCT'] >= change_FOLLOWED_RECOMMENDATIONS_PCT_low, original_df['FOLLOWED_RECOMMENDATIONS_PCT'] < change_FOLLOWED_RECOMMENDATIONS_PCT_hi)]
original_df['change_FOLLOWED_RECOMMENDATIONS_PCT'].replace(to_replace = condition,
                                                value      = 1,
                                                inplace    = True)

#Setting the trend change for CANCELLATIONS_BEFORE_NOON. DIDN'T HAVE ANY EFFECT ON AUC, it reduced the gap between Train and test
change_CANCELLATIONS_BEFORE_NOON = 2
original_df['change_CANCELLATIONS_BEFORE_NOON'] = 0
condition = original_df.loc[:,'change_CANCELLATIONS_BEFORE_NOON'][original_df['CANCELLATIONS_BEFORE_NOON'] < change_CANCELLATIONS_BEFORE_NOON]
original_df['change_CANCELLATIONS_BEFORE_NOON'].replace(to_replace = condition,
                                                value      = 1,
                                                inplace    = True)

#Setting the trend change for AVG_PREP_VID_TIME
change_AVG_PREP_VID_TIME = 253
original_df['change_AVG_PREP_VID_TIME'] = 0
condition = original_df.loc[:,'change_AVG_PREP_VID_TIME'][original_df['AVG_PREP_VID_TIME'] < change_AVG_PREP_VID_TIME]
original_df['change_AVG_PREP_VID_TIME'].replace(to_replace = condition,
                                                value      = 1,
                                                inplace    = True)

#Setting the trend change for TOTAL_PHOTOS_VIEWED. SMALL IMRPOVEMENT ON AUC
change_TOTAL_PHOTOS_VIEWED_low = 78
change_TOTAL_PHOTOS_VIEWED_hi  = 643
original_df['change_TOTAL_PHOTOS_VIEWED'] = 0
condition = original_df.loc[:,'change_TOTAL_PHOTOS_VIEWED'][np.logical_and(original_df['TOTAL_PHOTOS_VIEWED'] >= change_TOTAL_PHOTOS_VIEWED_low, original_df['TOTAL_PHOTOS_VIEWED'] < change_TOTAL_PHOTOS_VIEWED_hi)]
original_df['change_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition,
                                                value      = 1,
                                                inplace    = True)

#Setting the trend change for LARGEST_ORDER_SIZE. DIDN'T HAVE ANY EFFECT ON AUC, it reduced the difference between Train and test
change_LARGEST_ORDER_SIZE = 3
original_df['change_LARGEST_ORDER_SIZE'] = 0
condition = original_df.loc[:,'change_LARGEST_ORDER_SIZE'][original_df['LARGEST_ORDER_SIZE'] < change_LARGEST_ORDER_SIZE]
original_df['change_LARGEST_ORDER_SIZE'].replace(to_replace = condition,
                                                value      = 1,
                                                inplace    = True)

################################################################################
# Train/Test Split
################################################################################

# use this space to set up testing and validation sets using train/test split

# Note: Be sure to set test_size = 0.25
x_label = [
        'REVENUE'                            , 'CONTACTS_W_CUSTOMER_SERVICE'        , 'MOBILE_NUMBER',
        'CANCELLATIONS_BEFORE_NOON'          , 'CANCELLATIONS_AFTER_NOON'           , 'MOBILE_LOGINS',
        'WEEKLY_PLAN'                        , 'EARLY_DELIVERIES'                   , 'REFRIGERATED_LOCKER',
        'FOLLOWED_RECOMMENDATIONS_PCT'       , 'MEDIAN_MEAL_RATING'                 , 'AVG_CLICKS_PER_VISIT',
        'TOTAL_PHOTOS_VIEWED'                , 'IS_VALID_EMAIL'                     , 'change_FOLLOWED_RECOMMENDATIONS_PCT',
        'change_CANCELLATIONS_BEFORE_NOON'   , 'change_AVG_PREP_VID_TIME'           , 'change_TOTAL_PHOTOS_VIEWED'         ,
        'change_LARGEST_ORDER_SIZE'
    ]

y_label = ["CROSS_SELL_SUCCESS"]

X_data = original_df.loc[:, x_label]
y_data = original_df.loc[:, y_label]


scaler = StandardScaler()
scaler.fit(X_data)
X_scaled = scaler.transform(X_data)

X_train, X_test, y_train, y_test = train_test_split(X_scaled,
                                                    y_data,
                                                    test_size    = TEST_SIZE,
                                                    random_state = SEED,
                                                    stratify     = y_data)

################################################################################
# Final Model (instantiate, fit, and predict)
################################################################################

# use this space to instantiate, fit, and predict on your final model


model = LogisticRegression()

model.fit(X_train, y_train)
    
# Compute predicted probabilities: y_pred_prob
pred_y = model.predict(X_test)

#Train Score
train_score = model.score(X_train, y_train).round(3)

#Test Score
test_score  = model.score(X_test, y_test).round(3)    

#AUC Score
auc         = roc_auc_score(y_true  = y_test,
                            y_score = pred_y).round(3)


################################################################################
# Final Model Score (score)
################################################################################

# use this space to score your final model on the testing set
# MAKE SURE TO SAVE YOUR TEST SCORE AS test_score
# Example: test_score = final_model.score(X_test, y_test)

test_score = auc
print(auc)

