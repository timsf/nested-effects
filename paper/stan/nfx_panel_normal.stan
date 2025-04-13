data {

    int<lower=1> n_periods;
    int<lower=1> n_leaves;
    int<lower=1> n_covariates;
    int<lower=1> n_nodes_3;
    int<lower=1> n_nodes_2;
    int<lower=1> n_nodes_1;
    real<lower=0> prior_df;
    real<lower=0> prior_shape;
    real<lower=0> prior_rate;

    int<lower=1, upper=n_nodes_3> parent_node_3[n_leaves];
    int<lower=1, upper=n_nodes_2> parent_node_2[n_nodes_3];
    int<lower=1, upper=n_nodes_1> parent_node_1[n_nodes_2];

    matrix[n_covariates, n_periods] covariates;
    matrix[n_leaves, n_periods] response;
}


parameters {

    cov_matrix[n_covariates] cov_4;
    cov_matrix[n_covariates] cov_3;
    cov_matrix[n_covariates] cov_2;
    cov_matrix[n_covariates] cov_1;

    matrix[n_nodes_3, n_covariates] loc_3_raw;
    matrix[n_nodes_2, n_covariates] loc_2_raw;
    matrix[n_nodes_1, n_covariates] loc_1_raw;
    vector[n_covariates] loc_0;
    
    vector<lower=0>[n_leaves] var_resid;

    matrix[n_leaves, n_covariates] coefs_raw;
}


transformed parameters {
    
    cholesky_factor_cov[n_covariates] cf_cov_4 = cholesky_decompose(cov_4);
    cholesky_factor_cov[n_covariates] cf_cov_3 = cholesky_decompose(cov_3);
    cholesky_factor_cov[n_covariates] cf_cov_2 = cholesky_decompose(cov_2);
    cholesky_factor_cov[n_covariates] cf_cov_1 = cholesky_decompose(cov_1);

    matrix[n_nodes_1, n_covariates] loc_1 = rep_matrix(loc_0', n_nodes_1) + loc_1_raw * cf_cov_1';
    matrix[n_nodes_2, n_covariates] loc_2 = loc_1[parent_node_1] + loc_2_raw * cf_cov_2';
    matrix[n_nodes_3, n_covariates] loc_3 = loc_2[parent_node_2] + loc_3_raw * cf_cov_3';
    matrix[n_leaves, n_covariates] coefs = loc_3[parent_node_3] + coefs_raw * cf_cov_4';
    
    vector<lower=0>[n_leaves] scale_resid = sqrt(var_resid);
}


model {

    cov_4 ~ inv_wishart(prior_df, diag_matrix(rep_vector(prior_df, n_covariates)));
    cov_3 ~ inv_wishart(prior_df, diag_matrix(rep_vector(prior_df, n_covariates)));
    cov_2 ~ inv_wishart(prior_df, diag_matrix(rep_vector(prior_df, n_covariates)));
    cov_1 ~ inv_wishart(prior_df, diag_matrix(rep_vector(prior_df, n_covariates)));

    to_vector(loc_3_raw) ~ std_normal();
    to_vector(loc_2_raw) ~ std_normal();
    to_vector(loc_1_raw) ~ std_normal();
    to_vector(coefs_raw) ~ std_normal();
    
    var_resid ~ inv_gamma(prior_shape, prior_rate);

    to_vector(response) ~ normal(to_vector(coefs * covariates), to_vector(rep_matrix(scale_resid, n_periods)));
}
