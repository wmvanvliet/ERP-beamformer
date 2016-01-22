function W = lcmv_fit(X, template, varargin)
% Fit an LCMV Spatial beamformer.
% 
% Required parameters
% -------------------
% X : 3D matrix (n_channels x n_samples x n_trials)
%     The trials.  
%
% template : column or row vector (1 x n_channels | n_channels x 1)
%     Spatial activation pattern of the component to extract.
%
% Outputs
% -------
% W : row vector (1 x n_channels)
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
%     Whether to remove the channel mean before fitting the filter.

% Ensure template is a column vector
template = reshape(template, [], 1);

% Parse key/value parameters
p = inputParser;
addOptional(p, 'shrinkage', 'lw');
addOptional(p, 'center', false);
parse(p, varargin{:});
options = p.Results;

if options.center
    template = template - mean(template);
    X = X - repmat(mean(X, 1), size(X, 1), 1, 1));
end

% Concatenate trials
cont_eeg = reshape(permute(X, [1, 3, 2]), size(X, 1), []);

% Calculate spatial covariance matrix
if isfloat(options.shrinkage)
    spat_cov = cov(cont_eeg');
    spat_cov = (1 - options.shrinkage) * spat_cov * ...
               options.shrinkage * (trace(spat_cov) / size(spat_cov, 1)) * ...
               eye(size(spat_cov, 1));
elseif strcmp(options.shrinkage, 'lw')
    % Leodit-Wolf shrinkage
    spat_cov = cov1para(cont_eeg');
elseif strcmp(options.shrinkage, 'none')
    spat_cov = cov(cont_eeg');
end
 
sigma_x_i = pinv(spat_cov);

% Compute spatial LCMV filter
W = sigma_x_i * template;

% Noise normalization
W = W * inv(template' * sigma_x_i * template);

end
