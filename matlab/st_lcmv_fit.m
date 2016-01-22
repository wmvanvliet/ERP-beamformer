function W = st_lcmv_fit(X, template, varargin)
% Fit a spatio-temporal LCMV beamformer.
% 
% Required parameters
% -------------------
% X : 3D matrix (n_channels x n_samples x n_trials)
%     The trials.  
%
% template : 2D matrix (n_channels x n_samples)
%     Spatio-temporal activation pattern of the component to extract.
%
% Outputs
% -------
% W : row vector (1 x (n_channels * n_samples))
%   The filter weights of the beamformer.
% 
% Optional Parameters
% (specify as: 'key1', value1, 'key2', value2, ...)
% -------------------
% shrinkage : string or float (default: 'lw')
%     Shrinkage parameter for the covariance matrix inversion. This can
%     either be speficied as a number between 0 and 1, or as a string
%     indicating which automated estimation method to use:
% 
%     'none': No shrinkage: emperical covariance
%     'lw': Ledoit-Wolf approximation shrinkage
% 
% center : bool (default: true)
%     Whether to remove the data mean before fitting the filter.

% Parse key/value parameters
p = inputParser;
addOptional(p, 'shrinkage', 'lw');
addOptional(p, 'center', false);
parse(p, varargin{:});
options = p.Results;

if options.center
    template = _center(template);
    X = _center(X);
end

n_samples = size(X, 2);
template = template(:, 1:n_samples);

% Concatenate channels
trials = reshape(X, [], size(X, 3);

% Calculate spatial covariance matrix
if isfloat(options.shrinkage)
    spat_cov = cov(trials');
    spat_cov = (1 - options.shrinkage) * spat_cov * ...
               options.shrinkage * (trace(spat_cov) / size(spat_cov, 1)) * ...
               eye(size(spat_cov, 1));
elseif strcmp(options.shrinkage, 'lw')
    % Leodit-Wolf shrinkage
    spat_cov = cov1para(trials');
elseif strcmp(options.shrinkage, 'none')
    spat_cov = cov(trials');
end
 
sigma_x_i = pinv(spat_cov);

% Compute spatial LCMV filter
template = reshape(template, [], 1);
W = sigma_x_i * template;

% Noise normalization
W = W * inv(template' * sigma_x_i * template);

end

function X_trans = _center(X)
    data_mean = mean(reshape(X, [], size(X, 3)), 2);
    data_mean = reshape(data_mean, size(X, 1), size(X, 2), 1);
    X_trans = X - repmat(data_mean, size(X, 1), 1, 1);
end
