function X_trans = st_lcmv_apply(X, W, varargin)
% Apply a spatio-temporal LCMV beamformer to the data.
% 
% Required parameters
% -------------------
% X : 3D matrix (n_channels x n_samples x n_trials)
%     The trials.  
% W : row vector (1 x (n_channels * n_sampels))
%     The filter weights, obtained through the st_lcmv_fit function.
%
% Outputs
% -------
% X_trans : row vector (1 x n_trials)
%     The extracted timecourse.
%
% Optional Parameters
% (specify as: 'key1', value1, 'key2', value2, ...)
% -------------------
% center : bool (default: true)
%     Whether to remove the data mean before applying the filter.

% Parse key/value parameters
p = inputParser;
addOptional(p, 'center', false);
parse(p, varargin{:});
options = p.Results;

if options.center
    X = _center(X);
end

n_trials = size(X, 3);
X_trans = W' * reshape(X, [], n_trials))

end

function X_trans = _center(X)
    data_mean = mean(reshape(X, [], size(X, 3)), 2);
    data_mean = reshape(data_mean, size(X, 1), size(X, 2), 1);
    X_trans = X - repmat(data_mean, size(X, 1), 1, 1);
end
