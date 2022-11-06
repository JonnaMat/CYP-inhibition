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

svc_0 = BayesianOptimization(
    model=SVC,
    file_name=f"{task}/mcc/svc_0", 
    model_params=[
        Real(name="C", low=0.1, high=4.0)
    ],
    fix_model_params={"class_weight": "balanced"},
    datasets=datasets,
    feature_groups=feature_groups,
    main_metric="mcc"
)
[1.5198798389940014,
 0.0021614570564925,
 0.0129857353885148,
 0.0001530737007939,
 0.9586706863658084]

\begin{tabular}{lr}
\toprule
{} &        19 \\
\midrule
C                         &  1.519880 \\
var\_threshold\_continuous  &  0.002161 \\
var\_threshold\_discrete    &  0.012986 \\
var\_threshold\_fingerprint &  0.000153 \\
corr\_threshold            &  0.958671 \\
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