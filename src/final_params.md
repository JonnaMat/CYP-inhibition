# CYP2C9

xgboost_0 = BayesianOptimization(
    model=XGBClassifier,
    file_name=f"{task}/xgboost_0",
    model_params=[
        Integer(name="max_depth", low=5, high=50),
        Real(name="eta", low=0.01, high=0.2),
        Real(name="subsample", low=0.5, high=1),
        Real(name="scale_pos_weight", low=max(1, n_negative / n_positive - 1), high=6),
        Real(name="colsample_bytree", low=0.5, high=1.0),
        Real(name="lambda", low=0.5, high=4.0),
    ],
    fix_model_params={"objective": "binary:logistic", "eval_metric": "aucpr"},
    datasets=datasets,
    feature_groups=feature_groups,
    main_metric="mcc",
)

[35,
 0.1146619085062431,
 0.7143553768792537,
 3.293693458151421,
 0.9162053013568424,
 0.80223146109134,
 0.0214121340703628,
 0.0127910603984907,
 0.0225428454717947,
 0.8145397209176102]

\begin{tabular}{lr}
\toprule
{} &         33 \\
\midrule
max\_depth                 &  35 \\
eta                       &   0.114662 \\
subsample                 &   0.714355 \\
scale\_pos\_weight          &   3.293693 \\
colsample\_bytree          &   0.916205 \\
lambda                    &   0.802231 \\
var\_threshold\_continuous  &   0.021412 \\
var\_threshold\_discrete    &   0.012791 \\
var\_threshold\_fingerprint &   0.022543 \\
corr\_threshold            &   0.814540 \\
\bottomrule
\end{tabular}

# CYP1A2


catboost_0 = BayesianOptimization(
    model=CatBoostClassifier,
    file_name=f"{task}/catboost_0",
    model_params=[
        Integer(name="max_depth", low=4, high=12),
        Real(name="l2_leaf_reg", low=2., high=10.),
        # Categorical(name="boosting_type", categories=["Ordered", "Plain"]),
        Real(name="scale_pos_weight",low=max(1,n_negative/n_positive-1), high=6),
    ],
    fix_model_params={
        "verbose": 0,
    },
    datasets=datasets,
    feature_groups=feature_groups,
    main_metric="mcc",
)

[5,
 2.302130661080785,
 1.4119732352137206,
 0.0452883422715576,
 0.0464311306167034,
 0.007660577822017,
 0.8421726478843109]

\begin{tabular}{lr}
\toprule
{} &        40 \\
\midrule
max\_depth                 &  5 \\
l2\_leaf\_reg               &  2.302131 \\
scale\_pos\_weight          &  1.411973 \\
var\_threshold\_continuous  &  0.045288 \\
var\_threshold\_discrete    &  0.046431 \\
var\_threshold\_fingerprint &  0.007661 \\
corr\_threshold            &  0.842173 \\
\bottomrule
\end{tabular}

# CYP2D6

xgboost_0 = BayesianOptimization(
    model=XGBClassifier,
    file_name=f"{task}/mcc/xgboost_3",
    model_params=[
        Integer(name="max_depth", low=5, high=50),
        Real(name="eta", low=0.01, high=0.2),
        Real(name="subsample", low=0.5, high=1),
        Real(name="scale_pos_weight", low=max(1, n_negative / n_positive - 1), high=10),
        Real(name="colsample_bytree", low=0.5, high=1.0),
        Real(name="lambda", low=0.5, high=4.0),
    ],
    fix_model_params={"objective": "binary:logistic", "eval_metric": "aucpr"},
    datasets=datasets,
    feature_groups=feature_groups,
    main_metric="mcc",
)

[17,
 0.1184769179982453,
 0.9161494094487443,
 8.24730664960802,
 0.9735210994685166,
 3.3115968387668127,
 0.0016424271418347,
 0.0032084703916408,
 0.0125493960276156,
 0.9858382714381372]

\begin{tabular}{lr}
\toprule
{} &         43 \\
\midrule
max\_depth                 &  17.000000 \\
eta                       &   0.118477 \\
subsample                 &   0.916149 \\
scale\_pos\_weight          &   8.247307 \\
colsample\_bytree          &   0.973521 \\
lambda                    &   3.311597 \\
var\_threshold\_continuous  &   0.001642 \\
var\_threshold\_discrete    &   0.003208 \\
var\_threshold\_fingerprint &   0.012549 \\
corr\_threshold            &   0.985838 \\
\bottomrule
\end{tabular}

# CYP3A4

catboost_0 = BayesianOptimization(
    model=CatBoostClassifier,
    file_name=f"{task}/catboost_0",
    model_params=[
        Integer(name="max_depth", low=4, high=12),
        Real(name="l2_leaf_reg", low=2., high=10.),
        # Categorical(name="boosting_type", categories=["Ordered", "Plain"]),
        Real(name="scale_pos_weight",low=max(1,n_negative/n_positive-1), high=6),
    ],
    fix_model_params={
        "verbose": 0,
    },
    datasets=datasets,
    feature_groups=feature_groups,
    main_metric="mcc",
)

\begin{tabular}{lr}
\toprule
{} &        46 \\
\midrule
max\_depth                 &  9.000000 \\
l2\_leaf\_reg               &  3.951761 \\
scale\_pos\_weight          &  1.535843 \\
var\_threshold\_continuous  &  0.006960 \\
var\_threshold\_discrete    &  0.034635 \\
var\_threshold\_fingerprint &  0.021205 \\
corr\_threshold            &  0.875925 \\
\bottomrule
\end{tabular}


 [9.0,
 3.951761405575213,
 1.5358434365143534,
 0.0069601489987666,
 0.0346349723122983,
 0.0212049898683141,
 0.8759253469625337]

# CYP2C19

svc_0 = BayesianOptimization(
    model=SVC,
    file_name=f"{task}/svc_0", 
    model_params=[
        Real(name="C", low=0.1, high=4.0)
    ],
    fix_model_params={"class_weight": "balanced"},
    datasets=datasets,
    feature_groups=feature_groups,
    main_metric="mcc"
)

\begin{tabular}{lr}
\toprule
{} &        12 \\
\midrule
C                         &  1.306832 \\
var\_threshold\_continuous  &  0.016017 \\
var\_threshold\_discrete    &  0.027299 \\
var\_threshold\_fingerprint &  0.002842 \\
corr\_threshold            &  0.972394 \\
\bottomrule
\end{tabular}

[1.306832476304444,
 0.0160173437456012,
 0.027299499634864,
 0.0028418657614325,
 0.972393690403298]