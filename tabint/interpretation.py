import pdpbox
from pdpbox import pdp, info_plots
from .utils import *
from .pre_processing import *
from .learner import *
from matplotlib.colors import LinearSegmentedColormap
from .dataset import *
import graphviz
import shap
import shap.plots.colors as cl
from treeinterpreter import treeinterpreter as ti
import numpy as np


class PartialDependence:
    """
    Partial dependence https://github.com/SauceCat/PDPbox
    """
    def __init__(self, md, df, features, target):
        self.md = md
        self.df = df
        self.features = features
        self.target = target
        self.summary = {}
        
    @classmethod    
    def from_Learner(cls, learner, ds):
        df = ds.x_trn.copy()
        features = df.columns
        
        if len(ds.y_trn.shape) == 1:
            df['target'] = ds.y_trn
            target = ['target']
        else:
            target = []
            for i in range(ds.y_trn.shape.shape[1]):
                tgt_name = 'target' + str(i)
                df[tgt_name] = ds.y_trn.iloc[:,i]
                target.append(tgt_name)
        return cls(learner.md, df, features, target)
    
    def info_target_plot(self, feature, sample = 10000, target = None, grid_type = 'percentile', **kargs):
        fig, axes, result = info_plots.target_plot(
                df=self.sample(sample), feature=feature, feature_name=feature, 
                target=target or self.target, grid_type = grid_type, **kargs)
        self.info_target_data =  ResultDF(result, 'count')

        _ = axes['bar_ax'].set_xticklabels(self.summary['info_target'].display_column.values)
        plt.show()    
    
    def info_actual_plot(self, feature, sample = 10000, predict_kwds = {}, which_classes=None, **kargs):
        fig, axes, result = info_plots.actual_plot(
                model=self.md, 
                X=self.sample(sample), 
                feature=feature, feature_name=feature,
                predict_kwds=predict_kwds, which_classes = which_classes, **kargs)
        self.info_actual_data =  ResultDF(result, 'count')
        plt.show()        
    
    def isolate_plot(self, feature, sample = 10000,
                num_grid_points=10, grid_type='percentile',
                center = True, plot_lines=True, frac_to_plot=100, plot_pts_dist=True, 
                cluster=True, n_cluster_centers=10, cluster_method='accurate',
                which_classes= None, **pdp_kargs):
        ft_plot = pdp.pdp_isolate(
                model=self.md, dataset=self.sample(sample), 
                model_features = self.features, feature=feature,
                num_grid_points=num_grid_points, grid_type=grid_type,
                n_jobs=-1, **pdp_kargs)

        fig, axes = pdp.pdp_plot(
                pdp_isolate_out=ft_plot, feature_name=feature,
                center=center, plot_lines=plot_lines, frac_to_plot=frac_to_plot, plot_pts_dist=plot_pts_dist, 
                cluster=cluster, n_cluster_centers=n_cluster_centers, which_classes=which_classes)
        plt.show()
        
    def target_interact_plot(self, feature, var_name = None, target=None, sample = 10000, show_outliers=True, **kargs):
        fig, axes, self.summary['target_interact'] = info_plots.target_plot_interact(
                df=self.sample(sample), target= target or self.target,
                features= feature, feature_names = var_name or feature,
                show_outliers=show_outliers, **kargs)
        plt.show()
        
    def actual_interact_plot(self, feature, var_name = None, sample = 10000, which_classes = None, show_outliers=True, **kargs):
        fig, axes, result = info_plots.actual_plot_interact(
                model = self.md, X = self.sample(sample),
                features=feature, feature_names=var_name or feature, 
                which_classes=which_classes, show_outliers= show_outliers, **kargs)
        self.actual_interact_data =  ResultDF(result, 'count')
        plt.show()
        
    def pdp_interact_plot(self, feature, var_name=None, sample = 10000, which_classes = None,
                     num_grid_points=[10, 10], plot_types = None, plot_params = {'cmap': ["#00cc00", "#002266"]}):        
        ft_plot = pdp.pdp_interact(
                model=self.md, dataset=self.sample(sample), 
                model_features=self.features, features=feature, 
                num_grid_points=num_grid_points, n_jobs=4)
        
        plot_types = ['contour', 'grid'] if plot_types is None else [plot_types]
        for plot_type in plot_types:
            figs, ax = pdp.pdp_interact_plot(
                pdp_interact_out = ft_plot, 
                feature_names = var_name or feature, 
                plot_type= plot_type, plot_pdp=True, 
                which_classes=which_classes, plot_params = plot_params)
        plt.show()
    
    def sample(self, sample): return self.df if sample is None else self.df.sample(sample)



#harcode to change shap color
green_blue = LinearSegmentedColormap.from_list('custom blue', [(0, '#ffff00'), (1, '#002266')], N=256)
cl.red_blue = green_blue
cl.red_blue_solid = green_blue

class Shapley:
    """
    SHAP value: https://github.com/slundberg/shap
    """
    def __init__(self, explainer, shap_values, df, features):
        shap.initjs()
        self.explainer = explainer
        self.shap_values = shap_values
        self.df, self.features = df, features        
    
    @classmethod
    def from_Tree(cls, learner, ds, sample = 10000):
        df = ds.remove_outlier(inplace = False)[0]
        if sample < df.shape[0]: df = df.sample(sample)
        for c, v in df.items(): 
            if v.dtypes.name[:3] == 'int': df[c] = df[c].astype(np.float32)
        
        explainer = shap.TreeExplainer(learner.md)
        shap_values = explainer.shap_values(df)
        features = df.columns
        return cls(explainer, shap_values, df, features)

    @classmethod
    def from_kernel(cls): None

    def one_force_plot(self, loc = None, record = None, link='identity', plot_cmap = ["#00cc00", "#002266"]):
        s_values = self.shap_values[loc] if loc is not None else self.explainer.shap_values(record)[0]
        col_value = self.df.iloc[[loc]].values if loc is not None else record.values
        result = pd.DataFrame({'feature': self.features, 'feature value': col_value[0], 'Shap value': s_values})
        self.one_force_data = ResultDF(result, 'Shap value')
        return shap.force_plot(self.explainer.expected_value, s_values, features = self.features, plot_cmap = plot_cmap, link = link)
    
    def many_force_plot(self, loc, sample = 10000, plot_cmap = ["#00cc00", "#002266"]):
        return shap.force_plot(self.explainer.expected_value, self.shap_values[:loc,:], features = self.features, plot_cmap = plot_cmap)
    
    def summary_plot(self, plot_type = 'violin', alpha=0.3):
        """violin, layered_violin, dot"""
        return shap.summary_plot(self.shap_values, self.df, alpha=alpha, plot_type = plot_type)

    def importance_plot(self):
        return shap.summary_plot(self.shap_values, self.df, plot_type="bar")
        
    def interaction_plot(self, sample = 100):
        self.interaction_values = self.explainer.shap_interaction_values(self.df.sample(sample))
        return shap.summary_plot(self.interaction_values, features = self.features)
    
    def dependence_plot(self, col1, col2 = 'auto', alpha = 0.3, dot_size=50):
        return shap.dependence_plot(ind = col1, interaction_index = col2, 
                                    shap_values = self.shap_values, features = self.df, 
                                    alpha = alpha, dot_size=dot_size)


class Shapley_approx:
    """https://link.springer.com/article/10.1007/s10115-013-0679-x"""
    def __init__(self, shap_values, features, data):
        self.features = features
        self.shap_values =  shap_values
        self.data = data

    @classmethod
    def from_ds(cls, ds):
        #wip
        features = ds.features
        return cls(features)

    @classmethod
    def from_sequence(cls, learner, seq_df, instance, n_sample, features):
        df_sample = numpy_sample(seq_df, n_sample, 1)
        features_permute = np.array([np.random.permutation(i) for i in np.tile(np.array(list(range(len(features)))),(n_sample,1))])
        shap_values = cal_shap(learner, df_sample, instance, features, features_permute, n_sample)
        data = pd.DataFrame({'feature': features, 'shap_values': shap_values})
        return cls(shap_values, features, ResultDF(data, 'shap_values'))

    def construct_seq_instances(self, df_sample, instance, ith, features_permute):
        b1 = [np.concatenate([instance[:,:,:ith+1],df_sample[:,j:j+1,fts[ith+1:]]],axis=2) for j, fts in enumerate(features_permute)]
        b1 = np.concatenate(b1,axis=1)

        b2 = [np.concatenate([instance[:,:,:ith],df_sample[:,j:j+1,fts[ith:]]],axis=2) for j, fts in enumerate(features_permute)]
        b2 = np.concatenate(b2,axis=1)
        return b1, b2

    def cal_shap(self, learner, df_sample, instance, features, features_permute, n_sample):
        shap_values = []
        #to do nested for loop
        for i,f in enumerate(features):
            b1, b2 = construct_seq_instances(df_sample, instance, i, features_permute)
            phi = np.sum(learner.predict(b1) - learner.predict(b2))/n_sample
            shap_values.append(phi)
        return shap_values

    def plot(self, absolute=True, **kargs):
        plot_barh_from_series(self.features, self.shap_values, absolute=absolute, **kargs)



class Traterfall:
    def __init__(self, data):
        self.data = data
        
    @classmethod
    def from_df_loc(cls, learner, df, loc):
        return cls.from_record(learner, df.iloc[[loc]])

    @classmethod
    def from_record(cls, learner, record):
        prediction, bias, contributions = ti.predict(learner.md, record)
        contributions = [contributions[0][i][0] for i in range(len(contributions[0]))]
        data = pd.DataFrame({'feature': df.columns, 'value': record.values, 'contribute': contributions})
        return cls(ResultDF(data, 'contribute'))
        
    def plot(self, rotation_value=90, threshold=0.2, sorted_value=True, **kargs):
        my_plot = plot_waterfall(self.data().feature, self.data().contribute, rotation_value, threshold, sorted_value,**kargs)


class DrawTree:
    def __init__(self, es, features, tp):
        self.es = es
        self.features = features
        self.tp = tp
        
    @classmethod
    def from_SKLearn(cls, learner, ds, num_estimator = 0, size=10, ratio=0.6, precision=0):
        return cls(learner.md.estimators_, ds.features, 'SKTree')
    
    @classmethod
    def from_LGB(cls, learner): return cls(learner.md, None, 'LGB')
    
    @classmethod
    def from_XGB(cls): return None
    
    def plot(self, num_estimator = 0, **kargs): 
        if self.tp == 'SKTree': plot_SKTree(self.es[num_estimator], self.features, **kargs)
        elif self.tp =='LGB': plot_LGBTree(self.es, num_estimator, **kargs)
        elif self.tp == 'XGB': None