import numpy as np
import pandas as pd
import timeit
import matplotlib.pyplot as plt
import matplotlib as mpl
import json

from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial

from datetime import datetime
from dateutil import parser

from sklearn.preprocessing import MinMaxScaler
from math import e

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

scaler = MinMaxScaler()

def read_config():
    with open('config.json') as f:
        config = json.load(f)
    return config

# distribution by time and location
def disaggregate(data, var = 'date'):
	dates = []
	for n, row in data.iterrows():
		try:
			dates.append(datetime.strptime(row[var][:33], '%a %b %d %Y %H:%M:%S %Z%z'))
		except Exception as e:
			print(row[var])
			print(e)
			pass
	data['date'] = dates
	return data

def find_similarity(embedded_documents, reference, similarity = 'cosine'):
	start = timeit.default_timer()
	sims = []
	for embed in embedded_documents:
		sims.append(1-spatial.distance.cosine(reference, embed))
	stop = timeit.default_timer()
	print("time take for %d: %f" %(len(sims), (stop-start)))
	return sims

def plot_sim_dist(data, sustainability, simfield = '_sim_1'):
	fig, axs = plt.subplots(nrows=3, ncols=5, figsize = (16, 13), sharex = True, sharey = 'row')
	num = 0
	for i in range(4):
		for j in range(5):
			data[str(num)+simfield].sort_values().plot(kind = 'hist', ax = axs[i][j])
			axs[i][j].set_yscale('log')
			if j == 0:
				axs[i][j].set_ylabel("# reviews", fontsize = 16)
			else:
				axs[i][j].set_ylabel("", fontsize = 16)
			axs[i][j].set_xlabel("similarity", fontsize = 16)
			subtitle = sustainability['Goal'][num]
			if len(subtitle.split(' ')) > 3:
				midway = len(subtitle) // 2
				subtitle = subtitle[:midway] + "\n-" + subtitle[midway:] 
			axs[i][j].set_title("%s" %(subtitle), fontsize = 14)
			num += 1
		if num == 15:
			break
	

	for ax in axs.flatten():
		for tk in ax.get_yticklabels():
			tk.set_visible(True)
		for tk in ax.get_xticklabels():
			tk.set_visible(True)

	#plt.suptitle("Distribution of similaty scores with goals")
	#plt.tight_layout()
	plt.show()

def plot_sim_dist_(data, sustainability, which_goals, simfield = '_sim_1'):
	fig, axs = plt.subplots(nrows=3, ncols=5, figsize = (16, 13), sharex = True, sharey = False)
	num = 0
	for i in range(3):
		for j in range(5):
			data[str(which_goals[num])+simfield].sort_values().plot(kind = 'hist', ax = axs[i][j])
			axs[i][j].set_yscale('log')
			if j == 0:
				axs[i][j].set_ylabel("# reviews", fontsize = 16)
			else:
				axs[i][j].set_ylabel("", fontsize = 16)
			axs[i][j].set_xlabel("similarity", fontsize = 16)
			subtitle = sustainability['Goal'][which_goals[num]]
			if len(subtitle.split(' ')) > 3:
				midway = len(subtitle) // 2
				subtitle = subtitle[:midway] + "\n-" + subtitle[midway:] 
			axs[i][j].set_title("%s" %(subtitle), fontsize = 14)
			num += 1
			if num == len(which_goals):
				break
		if num == len(which_goals):
			break
	

	for ax in axs.flatten():
		for tk in ax.get_yticklabels():
			tk.set_visible(True)
		for tk in ax.get_xticklabels():
			tk.set_visible(True)

	#plt.suptitle("Distribution of similaty scores with goals")
	plt.tight_layout()
	plt.savefig("../plots/sim_dist.pdf")
    
def plot_sim_dist_odi(data, category_shorthands, construct, reference, reference_name, which_goals = [2], nrows = 3, ncols= 5, simfield = '_sim_1'):
	fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize = (16, 13), sharex = True, sharey = False)
	num = 0
	for i in range(nrows):
		for j in range(ncols):
			data['pros'][str(which_goals[num])+simfield].sort_values().plot(kind = 'hist',
                                                                            alpha = 0.4, label = "pros",
                                                                            ax = axs[i][j])
			data['cons'][str(which_goals[num])+simfield].sort_values().plot(kind = 'hist', alpha = 0.4,
                                                                            label = 'cons', ax = axs[i][j])
			threshold = data['cons'][str(which_goals[num])+simfield].quantile(0.95)
			axs[i][j].axvline(x=threshold, color = 'black', linestyle = 'dotted')            
			axs[i][j].set_yscale('log')
			axs[i][j].legend()
			if j == 0:
				axs[i][j].set_ylabel("# reviews", fontsize = 16)
			else:
				axs[i][j].set_ylabel("", fontsize = 16)
			axs[i][j].set_xlabel("similarity", fontsize = 16)
			subtitle = category_shorthands[construct][reference[reference_name][which_goals[num]]]
			if len(subtitle.split(' ')) > 3:
				midway = len(subtitle) // 2
				subtitle = subtitle[:midway] + "\n-" + subtitle[midway:] 
			axs[i][j].set_title("%s" %(subtitle), fontsize = 14)
			num += 1
			if num == len(which_goals):
				break
		if num == len(which_goals):
			break
	plt.legend()   
    
	
def subset_by_percentile(data, percentile, simfield = '_sim_1', num = 17):
	thresholded_data = []
	for num in range(0, num):
		threshold = data[str(num)+simfield].quantile(percentile)
		thresholded_data.append(data[data[str(num)+simfield] > threshold])
	return thresholded_data	
	
def subset_by_percentile_or_preset(data, percentile, simfield = '_sim_1', preset = 0.2, num = 17):
	thresholded_data = []
	for num in range(0, num):
		threshold = data[str(num)+simfield].quantile(percentile)
		data_ = data[data[str(num)+simfield] > preset]
		data_ = data_[data_[str(num)+simfield] > threshold]
		thresholded_data.append(data_)
	return thresholded_data	 

def subset_by_preset(data, simfield = '_sim_1', preset = 0.2):
    thresholded_data = []
    for num in range(0, 17):
        data_ = data[data[str(num)+simfield] > preset]
        thresholded_data.append(data_)
    return thresholded_data      

def sample_data(data, review_threshold = 1000):
	onek_companies = [i for i, j in data.groupby('company_name').size().\
					  sort_values(ascending = False).items() if j >= review_threshold]
	onek_data = data[data['company_name'].isin(onek_companies)]
	print("after sampling: ")
	print("# companies: ", len(onek_companies))
	print("Size of data: ", len(onek_data))
	print("Proportion of data discarded: ", (1 - len(onek_data) / len(data)))
	return onek_data

# if using a sentence based method
def aggregate_sents(data_90, main_data, sim = 'sim_1', num_goals = 17):
    subsetted_data = []
    # for a post with a shortlisted sentence, set it's score to be the maximum scoring sentence
    for g in range(num_goals):
        companies = data_90[g]["company_id"].values
        scores = data_90[g]["%d_%s" %(g, sim)].values
        post_scores = {}
        for n, i in enumerate(companies):
            post_scores[i] = -1

        for n, i in enumerate(companies):
            post_scores[i] = max(post_scores[i], scores[n])
    

        post_score = pd.DataFrame(post_scores.items(), columns = ["company_id", "%d_%s" %(g, sim)])
        subsetted_data.append(post_score.merge(main_data, on = "company_id"))
    return subsetted_data


def aggregate_area__(data, final_goals, onek_data, sustainability, area_name = 'company_name', reference_text = "Goal"):
    """
    Takes in an array of dataframes with shortlisted posts of different goals and returns a single dataframe
    aggregated by the area (company or state) with all 6 scores
    """
    by_goals = pd.DataFrame()
    area_list = {}
    posts = []
    horizontal_data = pd.DataFrame()
    for n, num in enumerate(final_goals):
        posts.append(data[num]['company_id'])
        area = pd.DataFrame(data[num].groupby(area_name).size(),
                                       columns = ['%d reviews' %num]).reset_index()
                
        area['avg sim score'] = list(data[num].groupby(area_name).mean()['%d_sim_1' %num].values)
        area_total_reviews = pd.DataFrame(onek_data.groupby(area_name).size(),
                                            columns = ['total']).reset_index()
        area = area.merge(area_total_reviews, on = area_name)
        area['%d reviews normalized' %(num)] = area['%d reviews' %(num)] / area['total']
        area['avg sim score'] = area['avg sim score'] / area['total']
        top5 = area.sort_values(['%d reviews normalized' %(num)], ascending = False)
        top5[reference_text] = [sustainability[reference_text][num]] * len(top5)
        top5.columns = [area_name, 'shortlisted reviews', '%d avg sim score' %(num),
                                'total reviews', '%d shortlisted prop' %(num), 'goal']
        by_goals = by_goals.append(top5, ignore_index=True)
        area_list[sustainability[reference_text][num]] = list(top5[area_name].values)
        
        if n == 0:
            horizontal_data = top5[[area_name, '%d shortlisted prop' %(num)]] #, '%d avg sim score' %(num)]] 
        else:
            horizontal_data = horizontal_data.merge(top5[[area_name, '%d shortlisted prop' %(num),
                                                              #'%d avg sim score' %(num),
                                                         ]],
                                                                on = area_name)
    
    return [by_goals, horizontal_data]


def aggregate_area_(data, final_goals, onek_data, sustainability,
                   metric = 'avg sim score',
                   area_name = 'company_name', scaled = False):
    """
    Takes in an array of dataframes with shortlisted posts of different goals and returns a single dataframe
    aggregated by the area (company or state) with all 6 scores
    """
    by_goals = pd.DataFrame()
    area_list = {}
    posts = []
    horizontal_data = pd.DataFrame()
    for n, num in enumerate(final_goals):
        posts.append(data[num]['company_id'])
        area = pd.DataFrame(data[num].groupby(area_name).size(),
                                       columns = ['%d reviews' %num]).reset_index()
                
        area['avg sim score'] = list(data[num].groupby(area_name).sum()['%d_sim_1' %num].values)
        area_total_reviews = pd.DataFrame(onek_data.groupby(area_name).size(),
                                            columns = ['total']).reset_index()
        area = area.merge(area_total_reviews, on = area_name)
        area['%d reviews normalized' %(num)] = area['%d reviews' %(num)] / area['total']
        area['avg sim score'] = area['avg sim score'] / area['total']
        top5 = area.sort_values(['%d reviews normalized' %(num)], ascending = False)
        top5['goal'] = [sustainability['Goal'][num]] * len(top5)
        top5.columns = [area_name, '%d shortlisted reviews' %(num), '%d avg sim score' %(num),
                                'total reviews', '%d pro proportion' %(num), 'goal']
        by_goals = by_goals.append(top5, ignore_index=True)
        area_list[sustainability['Goal'][num]] = list(top5[area_name].values)
        
        if scaled:
            top5['%d %s' %(num, metric)] = [i[0]*100 for i in list(scaler.fit_transform(top5[['%d %s' %(num, metric)]]))]
            
        
        if n == 0:
            horizontal_data = top5[[area_name, '%d %s' %(num, metric)]] #, '%d avg sim score' %(num)]] 
        else:
            horizontal_data = horizontal_data.merge(top5[[area_name, '%d %s' %(num, metric),
                                                              #'%d avg sim score' %(num),
                                                         ]],
                                                                on = area_name)
        
    # find composite score
    df_ = pd.DataFrame()
    for col in horizontal_data.columns:
        if horizontal_data[col].dtype in numerics:
            col_zscore = col + '_zscore'
            df_[col_zscore] = (horizontal_data[col] - horizontal_data[col].mean())/horizontal_data[col].std(ddof=0)
    horizontal_data['sustainability score'] = df_.sum(axis=1)
    return [by_goals, horizontal_data]    


def aggregate_area(data, final_goals, onek_data, sustainability,
                   metric = 'avg sim score',
                   construct = 'sustainability',
                   area_name = 'company_name', scaled = False,
                   scoring = 'normal',
                   composite = True,
                   find_pcs = True,
                   reference_text = 'Goal'):
    """
    Takes in an array of dataframes with shortlisted posts of different goals and returns a single dataframe
    aggregated by the area (company or state) with all 6 scores + composite score and PCs
    """
    by_goals = pd.DataFrame()
    area_list = {}
    posts = []
    horizontal_data = pd.DataFrame()
    for n, num in enumerate(final_goals):
        posts.append(data[num]['company_id'])
        area = pd.DataFrame(data[num].groupby(area_name).size(),
                                       columns = ['%d reviews' %num]).reset_index()

        # exponential sim
        data[num]['%d_sim_1_exp' %num] = np.exp(data[num]['%d_sim_1' %num])/e
        # log sim
        data[num]['%d_sim_1_log' %num] = np.log(data[num]['%d_sim_1' %num] + 1)/np.log(2)
        
        if scoring == 'normal':
            sim_metric = '%d_sim_1' %num
        elif scoring == 'exp':
            sim_metric = '%d_sim_1_exp' %num
        elif scoring == 'log':
            sim_metric = '%d_sim_1_log' %num
        
        area['avg sim score'] = list(data[num].groupby(area_name).sum()[sim_metric].values)
        area_total_reviews = pd.DataFrame(onek_data.groupby(area_name).size(),
                                            columns = ['total']).reset_index()
        area = area.merge(area_total_reviews, on = area_name)
        area['%d reviews normalized' %(num)] = area['%d reviews' %(num)] / area['total']
        area['avg sim score'] = area['avg sim score'] / area['total']
        top5 = area.sort_values(['%d reviews normalized' %(num)], ascending = False)
        top5[reference_text] = [sustainability[reference_text][num]] * len(top5)
        top5.columns = [area_name, '%d shortlisted reviews' %(num), '%d avg sim score' %(num),
                                'total reviews', '%d pro proportion' %(num), reference_text]
        by_goals = by_goals.append(top5, ignore_index=True)
        area_list[sustainability[reference_text][num]] = list(top5[area_name].values)
        
        if scaled:
            top5['%d %s' %(num, metric)] = [i[0]*100 for i in list(scaler.fit_transform(top5[['%d %s' %(num, metric)]]))]
        
        
        if n == 0:
            horizontal_data = top5[[area_name, '%d %s' %(num, metric)]] #, '%d avg sim score' %(num)]] 
        else:
            horizontal_data = horizontal_data.merge(top5[[area_name, '%d %s' %(num, metric),
                                                              #'%d avg sim score' %(num),
                                                         ]],
                                                                on = area_name)
        
    # find composite score
    if composite:
        df_ = pd.DataFrame()
        for col in horizontal_data.columns:
            if horizontal_data[col].dtype in numerics:
                col_zscore = col + '_zscore'
                df_[col_zscore] = (horizontal_data[col] - horizontal_data[col].mean())/horizontal_data[col].std(ddof=0)
        horizontal_data['composite'] = df_.sum(axis=1)
    

    # for odi, combine 1,5 (psychological) and 2,3 (physiological) 
    if construct == 'odi':
        df_ = pd.DataFrame()
        for col_ in [1, 5]:
            col = "%s avg sim score" %col_
            if horizontal_data[col].dtype in numerics:
                col_zscore = col + '_zscore'
                df_[col_zscore] = (horizontal_data[col] - horizontal_data[col].mean())/horizontal_data[col].std(ddof=0)
        horizontal_data['psychological'] = df_.sum(axis=1)
        horizontal_data['psychological'] = (horizontal_data['psychological'] - horizontal_data['psychological'].mean())/horizontal_data['psychological'].std(ddof=0)


        df_ = pd.DataFrame()
        for col_ in [2, 3]:
            col = "%s avg sim score" %col_
            if horizontal_data[col].dtype in numerics:
                col_zscore = col + '_zscore'
                df_[col_zscore] = (horizontal_data[col] - horizontal_data[col].mean())/horizontal_data[col].std(ddof=0)
        horizontal_data['physiological'] = df_.sum(axis=1)
        horizontal_data['physiological'] = (horizontal_data['physiological'] - horizontal_data['physiological'].mean())/horizontal_data['physiological'].std(ddof=0)


    # find PC's
    if find_pcs:
        from sklearn.decomposition import PCA
        pca = PCA(n_components = 3)
        feat_cols = ['%d %s'%(i, metric) for i in final_goals]
        pca_result = pca.fit_transform(horizontal_data[feat_cols].values)
        horizontal_data['PCA1'] = pca_result[:,0]
        horizontal_data['PCA2'] = pca_result[:,1] 

        horizontal_data['pca-one-rank'] = horizontal_data['PCA1'].rank(method='dense', ascending=False)
        horizontal_data['pca-two-rank'] = horizontal_data['PCA2'].rank(method='dense', ascending=False)
        horizontal_data['rank-diff'] = horizontal_data['pca-two-rank'] - horizontal_data['pca-one-rank']

        # scale the rank-diff

    return [by_goals, horizontal_data]


if __name__ == "__main__" :
	pass