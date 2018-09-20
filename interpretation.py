import pdpbox
from pdpbox import pdp, info_plots
from .learner import *

#pdp plot
#Accumulated Local Effect plots 
#https://christophm.github.io/interpretable-ml-book
#https://github.com/slundberg/shap


class PartialDependence():
    def __init__(self, md, df, features, target):
        self.md = md
        self.df = df
        self.features = features
        self.target = target
        self.summary = {}
        
    @classmethod    
    def from_LGBLearner(cls, learner):
        df = dataset.lgb_trn.data.copy()
        features = df.columns
        
        if len(dataset.lgb_trn.label.shape) == 1:
            df['target'] = dataset.lgb_trn.label
            target = ['target']
        else:
            target = []
            for i in range(dataset.lgb_trn.label.shape[1]):
                tgt_name = 'target' + str(i)
                df[tgt_name] = dataset.lgb_trn.label.iloc[:,i]
                target.append(tgt_name)
        return cls(learner.md, df, features, target)
    
    def info_target(self, var, sample = 10000, target = None, grid_type = 'percentile', **kargs):        
        fig, axes, self.summary['info_target'] = info_plots.target_plot(
                df=self.sample(sample), feature=var, feature_name=var, 
                target=isnone(target, self.target), grid_type = grid_type, **kargs)

        _ = axes['bar_ax'].set_xticklabels(self.summary['info_target'].display_column.values)
        plt.show()    
    
    def info_actual(self, var, sample = 10000, predict_kwds = {}, which_classes=None, **kargs):
        
        fig, axes, self.summary['info_actual'] = info_plots.actual_plot(
                model=self.md, 
                X=self.sample(sample), 
                feature=var, feature_name=var,
                predict_kwds=predict_kwds, which_classes = which_classes, **kargs)
        plt.show()        
    
    def isolate(self, var, sample = 10000,
                num_grid_points=10, grid_type='percentile',
                center = True, plot_lines=True, frac_to_plot=100, plot_pts_dist=True, 
                cluster=True, n_cluster_centers=10, cluster_method='accurate',
                which_classes= None, **pdp_kargs):
        
        ft_plot = pdp.pdp_isolate(
                model=self.md, dataset=self.sample(sample), 
                model_features = self.features, feature=var,
                num_grid_points=num_grid_points, grid_type=grid_type,
                n_jobs=-1, **pdp_kargs)

        fig, axes = pdp.pdp_plot(
                pdp_isolate_out=ft_plot, feature_name=var,
                center=center, plot_lines=plot_lines, frac_to_plot=frac_to_plot, plot_pts_dist=plot_pts_dist, 
                cluster=cluster, n_cluster_centers=n_cluster_centers, which_classes=which_classes)
        plt.show()
        
    def target_interact(self, var, var_name = None, target=None,
                        sample = 10000, show_outliers=True, **kargs):
    
        fig, axes, self.summary['target_interact'] = info_plots.target_plot_interact(
                df=self.sample(sample), target=isnone(target, self.target),
                features= var, feature_names = isnone(var_name, var),
                show_outliers=show_outliers, **kargs)
        plt.show()
        
    def actual_interact(self, var, var_name = None, 
                        sample = 10000, which_classes = None, show_outliers=True, **kargs):
    
        fig, axes, self.summary['actual_interact'] = info_plots.actual_plot_interact(
                model = self.md, X = self.sample(sample),
                features=var, feature_names=isnone(var_name, var), 
                which_classes=which_classes, show_outliers= show_outliers, **kargs)
        plt.show()
        
    def pdp_interact(self, var, var_name=None, sample = 10000, which_classes = None,
                     num_grid_points=[10, 10], plot_types = None):
        
        ft_plot = pdp.pdp_interact(
                model=self.md, dataset=self.sample(sample), 
                model_features=self.features, features=var, 
                num_grid_points=num_grid_points, n_jobs=4)
        
        plot_types = ['contour', 'grid'] if plot_types is None else [plot_types]
        for plot_type in plot_types:
            figs, ax = pdp.pdp_interact_plot(
                pdp_interact_out = ft_plot, 
                feature_names = isnone(var_name, var), 
                plot_type= plot_type, plot_pdp=True, which_classes=which_classes)
        plt.show()
    
    def sample(self, sample): return self.df if sample is None else self.df.sample(sample)