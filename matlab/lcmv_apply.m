function X_trans = lcmv_apply(X, W, varargin)
% Apply an LCMV Spatial beamformer to the data.
% 
% Required parameters
% -------------------
% X : 3D matrix (n_channels x n_samples x n_trials)
%     The trials.  
% W : row vector (1 x n_channels)
%     The filter weights, obtained through the lcmv_fit function.
%
% Outputs
% -------
% X_trans : 3D matrix (1 x samples x n_trials)
%     The extracted timecourses.
%
% Optional Parameters
% (specify as: 'key1', value1, 'key2', value2, ...)
% -------------------
% center : bool (default: true)
%     Whether to remove the channel mean before applying the filter.

% Parse key/value parameters
p = inputParser;
addOptional(p, 'center', false);
parse(p, varargin{:});
options = p.Results;

if options.center
    X = X - repmat(mean(X, 1), size(X, 1), 1, 1));
end

n_channels = size(W, 2);
n_samples = size(X, 2);
n_trials = size(X, 3);

X_trans = zeros(n_channels, n_samples, n_trials);
for i = 1:n_trials
    X_trans(:, :, i) = W' * X(:, :, i):
end

end
