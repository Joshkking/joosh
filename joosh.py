def cm_sim(n=100, propt=0.5, wtp=1, wtn=1, wfp=-1, wfn=-1):
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    #import seaborn as sb
    
    # proportion actual false
    propf = 1 - propt
    # count the number of actual trues and falses
    countt = int(propt * n)
    countf = int(propf * n)
    
    # create an array representing the actuals
    a = np.full((countt), 1)
    b = np.full((countf), 0)
    y = np.concatenate((a, b))
    
    # create an array representing probability steps of the proportion gotten wrong
    eprobs = np.arange(start=0, stop=1.05, step=0.05)
    
    # create some empty lists to store items in
    t_error_list = []
    f_error_list = []
    score_list = []
    tp_list = []
    tn_list = []
    fp_list = []
    fn_list = []
    
    # loop through each combination of error probabilities so we can have error in trues and falses
    for t_error in eprobs:
        for f_error in eprobs:
            
            # append the relevant error probabilities
            t_error_list.append(t_error)
            f_error_list.append(f_error)
            
            # resest yhat
            yhat = y.copy()
            
            # find how many trues we need to mess up
            tindex = int(countt * t_error)
            # change that many trues to falses
            if t_error != 0:
                yhat[:tindex] = 0
            
            # find how many falses we need to mess up
            findex = int(countf * f_error)
            # change that many falses to trues
            if f_error != 0:
                yhat[-findex:] = 1
            
            # reset (or create) the counters
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            
            # append a score based on whether the prediction was correct or not
            for i in range(len(y)):
                if y[i] == 1 and yhat[i] == 1:
                    tp += 1
                elif y[i] == 0 and yhat[i] == 0:
                    tn += 1
                elif y[i] == 0 and yhat[i] == 1:
                    fp += 1
                elif y[i] == 1 and yhat[i] == 0:
                    fn += 1
                    
            # calculate the score and append to the relevant lists
            score = (wtp)*tp + (wtn)*tn + (wfp)*fp + (wfn)*fn
            score_list.append(score)
            tp_list.append(tp)
            tn_list.append(tn)
            fp_list.append(fp)
            fn_list.append(fn)
    
    # Create a dataframe for easy housing the data
    df = pd.DataFrame(data={
        'Positive Class Error (%)': t_error_list
        , 'Negative Class Error (%)': f_error_list
        , 'Score':score_list
        , 'tp': tp_list
        , 'tn': tn_list
        , 'fp': fp_list
        , 'fn': fn_list
    })
    
    #plt.figure(figsize=(7,7))
    #sb.scatterplot(data=df, y='Positive Class Error (%)', x='Negative Class Error (%)'
    #               , hue='Score', size='Score', palette='viridis_r')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.show()
    
    # Plot the results
    plt.figure(figsize=(8, 7))
    plt.hexbin(x=df['Negative Class Error (%)'], y=df['Positive Class Error (%)'], C=df.Score, gridsize=15)
    plt.xlabel('Negative Class Error (%)')
    plt.ylabel('Positive Class Error (%)')
    cbar = plt.colorbar()
    cbar.set_label('Score')
    plt.show()




class stepwise:
    
    def __init__(self, data, target, features, method='forward', target_type='numerical', pcutval=0.05, checks=[]):
        
        # Import necessary libraries if not already done
        import pandas as pd
        import statsmodels.api as sm
        
        # parameters
        # data -> a pandas dataframe
        # target -> the target/independent variable (as a string)
        # target_type -> the datatype of the target
        # features -> the features to iterate over (as a list)
        # method -> the type of autoselection ('forward' or 'backward')
        # pcutval -> the p-value cutoff for adding/removing features
        # checks -> list of statistical checks to make at each iteration
        
        # Set some instance attributes if wanting to reference back
        self.target = target
        self.target_type = target_type
        self.features = features.copy() # We will mutate this so make sure playing with new object
        self.num_features = len(features)
        self.checks = ['pvalue'] + checks # Convenient way to force pvalue check in loop later on
        
        # Sort the features in order of correlation with target to make forward search slightly more intelligent
        self.corrs = data[self.features].corrwith(data[self.target], method='pearson')
        self.corrs = abs(self.corrs) # since correlations can be pos/neg
        self.corrs.sort_values(ascending=False, inplace=True) 
        self.features = list(self.corrs.index) # Finally update the feature list
        
        # Set up initial lists to append items to for each step
        self.step_data = [] # The iteration step
        self.current_feature = [] # The current feature being evaluated in the process
        self.features_used_history = [] # A history of features used (a list of lists)
        self.stat_checks = [] # A list capturing any failed statistics checks at each step
        self.delta_data = [] # List to elaborate the action at each step (i.e. feature added/not added/removed)
        self.ignore_p = [] # List of features that have bad p values, but to ignore them as dropping fails a stat check
         # The running list of all features to be used at each model step (for backwards, start with all)
        if method == 'forward':
            self.features_used_current = []
        elif method == 'backward':
            self.features_used_current = features.copy() # copy to make sure this points to a new object
        
        # Various lists to append the relevant statistic for each step
        self.pval_data = []
        self.R2_data = []
        self.AdjR2_data = []
        self.AIC_data = []
        self.BIC_data = []
        
        # Set up a few variables to use as part of the iterations
        self.i = 0 # Simple index var
        self.last_viable_step = 0 # The last index step producing a viable data (dependent on the method)
        self.break_check = False # As long as false, continue the main body iteration
        
        # Main iteration body
        # The check on i is an infinite loop protection. This should be avoided but just in case
        while (self.break_check == False) and (self.i < self.num_features * 3):

            # Find the current feature and append it to the necessary lists for forward selection
            if method == 'forward':
                feature = self.features[self.i]
                self.current_feature.append(feature) # The list for the eventual report
                self.features_used_current.append(feature) # The list for all features used in the model

            # Build the model with the current features
            model_string = target + ' ~ ' + ' + '.join(self.features_used_current)
            if self.target_type =='numerical':
                model = sm.formula.ols(formula=model_string, data=data)
                model_fitted = model.fit(disp=False)
            elif self.target_type =='categorical':
                model = sm.formula.logit(formula=model_string, data=data)
                model_fitted = model.fit(disp=False, method='bfgs', maxiter=1000)
            else:
                raise ValueError('target_type must be either "numerical"or "categorical"')


            # Find current relevant statistics
            # Rounding is needed for statcheck variables to deal with slight stochastics in far decimal points
            curr_AIC = round(model_fitted.aic, 5)
            curr_BIC = round(model_fitted.bic, 5)
            if target_type == 'numerical':
                curr_R2 = round(model_fitted.rsquared, 5)
                curr_AdjR2 = round(model_fitted.rsquared_adj, 5)
            else:
                # for logistic models we have to use pseuo R2
                curr_R2 = round(model_fitted.prsquared, 5)
                curr_AdjR2 = 'N/A' # statsmodels does not currently have this available for logit
            pvals = model_fitted.pvalues[1:].copy() # The slicing at 1 prevents including the intercept
            if method == 'backward': # for backwards, we must remove any pvals in our ignore list before finding worst
                for idx in pvals.index:
                    if idx in self.ignore_p:
                        pvals.drop(labels=idx, inplace=True)
            worst_pval_idx = pvals.values.argmax()
            worst_pval = pvals.max()
            worst_pfeature = pvals.index[worst_pval_idx]
            if method == 'forward': # for forwards we don't care about worst pvalue, only the one we are on
                current_pval = pvals.loc[feature]

            # Append data to relevant lists for eventual model report
            self.step_data.append(self.i)
            self.features_used_history.append(list(self.features_used_current)) # list ensures appending list object
            self.AIC_data.append(curr_AIC)
            self.BIC_data.append(curr_BIC)
            self.R2_data.append(curr_R2)
            self.AdjR2_data.append(curr_AdjR2)
            if method == 'forward':
                self.pval_data.append(current_pval)
            elif method == 'backward':
                self.pval_data.append(worst_pval)

            # Get the prior statistics values for use in any checks
            prev_AIC = self.AIC_data[self.last_viable_step]
            prev_BIC = self.BIC_data[self.last_viable_step]
            prev_R2 = self.R2_data[self.last_viable_step]
            prev_AdjR2 = self.AdjR2_data[self.last_viable_step]

            # Some helper flags/functions for the statistical checks
            p_check = 1 # defaulted to 1 to keep the current, or all, features (0 to remove)
            bstat_check = 0 # statistics check for backwards elim, if 1 a check failed (so don't remove the feature)
            curr_failures = [] # for keeping track of failures within each step
            def stat_fail(failure): # Quick helper function to fail the checks
                curr_failures.append(' Failure of {}'.format(failure))

            # The main logic for the various statistical checks

            for check in self.checks:

                if check == 'pvalue':
                    if method == 'forward' and current_pval >= pcutval:
                        stat_fail(check)
                        p_check = 0
                    elif method == 'backward' and worst_pval >= pcutval:
                        # Don't include statfail here since for backwards this isn't helpful info
                        p_check = 0
                elif check == 'AIC':
                    if curr_AIC > prev_AIC:
                        if method == 'forward':
                            stat_fail(check)
                            p_check = 0
                        elif method == 'backward':
                            stat_fail(check)
                            bstat_check = 1
                elif check == 'BIC':
                    if curr_BIC > prev_BIC:
                        if method == 'forward':
                            stat_fail(check)
                            p_check = 0
                        elif method == 'backward':
                            stat_fail(check)
                            bstat_check = 1
                elif check == 'R2':
                    if curr_R2 < prev_R2:
                        if method == 'forward':
                            stat_fail(check)
                            p_check = 0
                        elif method == 'backward':
                            raise ValueError('R2 is an invalid check for backwards selection. See docs.')
                elif check == 'AdjR2':
                    if self.target_type == 'categorical':
                        raise ValueError('AdjR2 is not a valid check type for logistic regression. See docs.')
                    elif curr_AdjR2 < prev_AdjR2:
                        if method == 'forward':
                            stat_fail(check)
                            p_check = 0
                        elif method == 'backward':
                            stat_fail(check)
                            bstat_check = 1
                else:
                    raise ValueError('An invalid statistical check was provided')

            # Finalize the decision logic

            if method == 'forward':

                if p_check == 0:
                    self.features_used_current.remove(feature)
                    self.delta_data.append('Not added')
                    self.stat_checks.append(list(curr_failures))
                    self.last_viable_step = self.i

                else:
                    self.delta_data.append('Added')
                    self.stat_checks.append(list(['No failures'])) # List to match forms
                    self.last_viable_step = self.i

            elif method == 'backward':

                if bstat_check == 1:
                    # In this case a statistical check failed
                    # As such, we want to reclaim the previously dropped feature (since we actually want it)
                    self.stat_checks.append(list(curr_failures))
                    previous_dropped =  self.delta_data[self.last_viable_step]
                    stat_notice = 'Worst pvalue is [{}] but stat check failed compared to previous run.\
                        No removals and reclaiming [{}] for next run.'.format(worst_pfeature, previous_dropped)
                    self.delta_data.append(stat_notice)
                    self.features_used_current.append(previous_dropped)
                    self.ignore_p.append(previous_dropped) # so we can ignore this feature's pvalue in the future

                elif p_check == 0:
                    self.features_used_current.remove(worst_pfeature)
                    self.delta_data.append(worst_pfeature)
                    self.stat_checks.append(list(['No failures']))
                    self.last_viable_step = self.i

                elif p_check == 1:
                    self.delta_data.append('Nothing Removed')
                    self.stat_checks.append(list(['No failures']))
                    self.last_viable_step = self.i

                else:
                    raise ValueError('Some error occured in the final decision logic for backwards elimination.')
            
            # Some print lines for debugging if necessary
#             print('i: {}'.format(self.i))
#             print('last viable i: {}'.format(self.last_viable_step))
#             print('bstat_check: {}'.format(bstat_check))
#             print('Delta data for this step:')
#             print(self.delta_data[self.i])
#             print('Features_used_current list:')
#             print(self.features_used_current)
#             print('Pvals of what was actually used in this model:')
#             print(model_fitted.pvalues)
#             print('Pvals that the p check had access to:')
#             print(pvals)
#             print('prev_AIC: {}'.format(prev_AIC))
#             print('prev_BIC: {}'.format(prev_BIC))
#             print('prev_AdjR2: {}'.format(prev_AdjR2))
#             print('curr_AIC: {}'.format(curr_AIC))
#             print('curr_BIC: {}'.format(curr_BIC))
#             print('curr_AdjR2: {}'.format(curr_AdjR2))
#             print('\n')
#             print('\n')
            
            # Increase the index counter
            self.i = self.i + 1
            
            # End the iteration if the relevant conditions are met
            if (method == 'backward') and (worst_pval < pcutval) and bstat_check != 1:
                # if no pvals above cut in backwards and we're not in the middle of a stat check fail
                self.break_check = True
            elif method == 'forward' and self.i == (self.num_features):
                # if the index has ran the total feature count
                self.break_check = True

        # Set the final attributes of the winning model
        self.final_predictors = self.features_used_current
        self.final_string = self.target + ' ~ ' + ' + '.join(self.features_used_current)
        if self.target_type =='numerical':
            self.final_model = sm.formula.ols(formula=self.final_string, data=data)
            self.final_fit = self.final_model.fit(disp=False)
        elif self.target_type =='categorical':
            self.final_model = sm.formula.logit(formula=self.final_string, data=data)
            self.final_fit = self.final_model.fit(disp=False, method='bfgs', maxiter=1000)
        else:
            raise ValueError('target_type must be either "numerical"or "categorical"')
        
        # Build a pandas dataframe as a report of the steps in the process
        if method == 'forward':
            self.stepdata_dict = {
                'Step': self.step_data,
                'Features Used': self.features_used_history,
                'Current Feature': self.current_feature,
                'Added': self.delta_data,
                'Stat Checks': self.stat_checks,
                'Current Pval': self.pval_data,
                'R2': self.R2_data,
                'AdjR2': self.AdjR2_data,
                'AIC': self.AIC_data,
                'BIC': self.BIC_data,
                }
        elif method == 'backward':
            self.stepdata_dict = {
                'Step': self.step_data,
                'Features Used': self.features_used_history,
                'Feature Removed': self.delta_data,
                'Stat Checks': self.stat_checks,
                'Worst Pval': self.pval_data,
                'R2': self.R2_data,
                'AdjR2': self.AdjR2_data,
                'AIC': self.AIC_data,
                'BIC': self.BIC_data,
                }
        
        self.step_df = pd.DataFrame(self.stepdata_dict)
  

    # Function to fit the final model summary
    def final_summary(self):
        print(self.final_fit.summary())
