classdef (StrictDefaults)MIMOChannel2 < CustomChannelBase
%MIMOChannel2 Filter input signal through a MIMO multipath fading channel
%   CHAN = comm.MIMOChannel2 creates a multiple-input multiple-output (MIMO)
%   frequency-selective or frequency-flat fading channel System object,
%   CHAN. This object filters a real or complex input signal through the
%   multipath MIMO channel to obtain the channel impaired signal.
%   comm.internal.
%   CHAN = comm.MIMOChannel2(Name,Value) creates a MIMO channel object,
%   CHAN, with the specified property Name set to the specified Value. You
%   can specify additional name-value pair arguments in any order as
%   (Name1,Value1,...,NameN,ValueN).
%
%   Step method syntax:
%
%   Y = step(CHAN,X) filters input signal X through a MIMO fading channel
%   and returns the result in Y. The input X can be a double precision data
%   type scalar, vector, or 2D matrix with real or complex values. X is of
%   size Ns x Nt, where Ns is the number of samples and Nt is the number of
%   transmit antennas that is determined by the NumTransmitAntennas or
%   TransmitCorrelationMatrix property value of CHAN. Y is the output
%   signal of size Ns x Nr, where Nr is the number of receive antennas that
%   is determined by the NumReceiveAntennas, ReceiveCorrelationMatrix or a
%   combination of SpatialCorrelationMatrix and NumTransmitAntennas
%   property values of CHAN. Y is of double precision data type with
%   complex values.
%  
%   Y = step(CHAN,X,SELTX) turns on selected transmit antennas for X
%   transmission. This syntax applies when you set the AntennaSelection
%   property of CHAN to 'Tx'. SELTX is a numeric type binary-valued 1 x Nt
%   row vector, in which the ones indicate the selected transmit antennas.
%   X is of size Ns x Nst, where Nst is the number of selected transmit
%   antennas, i.e., the number of ones in SELTX. Y is of size Ns x Nr.
%   
%   Y = step(CHAN,X,SELRX) turns on selected receive antennas for X
%   transmission. This syntax applies when you set the AntennaSelection
%   property of CHAN to 'Rx'. SELRX is a numeric type binary-valued 1 x Nr
%   row vector, in which the ones indicate the selected receive antennas. X
%   is of size Ns x Nt. Y is of size Ns x Nsr, where Nsr is the number of
%   selected receive antennas, i.e., the number of ones in SELRX.
% 
%   Y = step(CHAN,X,SELTX,SELRX) turns on selected transmit and receive
%   antennas for X transmission. This syntax applies when you set the
%   AntennaSelection property of CHAN to 'Tx and Rx'. X is of size Ns x
%   Nst, and Y is of size Ns x Nsr.
% 
%   Y = step(CHAN,...,INITIALTIME) filters input signal X through a MIMO
%   fading channel, with the fading process starting at INITIALTIME, and
%   returns the result in Y. INITIALTIME is a double precision, nonnegative
%   scalar measured in seconds. The product of INITIALTIME and the
%   'SampleRate' property of CHAN must be an integer. This syntax applies
%   when you set the 'FadingTechnique' property of CHAN to 'Sum of
%   sinusoids' and the 'InitialTimeSource' property of CHAN to 'Input
%   port'.
% 
%   [Y,PATHGAINS] = step(CHAN,X) returns the MIMO channel path gains of the
%   underlying fading process in PATHGAINS. This syntax applies when you
%   set the PathGainsOutputPort property of CHAN to true. PATHGAINS is of
%   size Ns x Np x Nt x Nr, where Np is the number of paths, i.e., the
%   length of the PathDelays property value of CHAN. PATHGAINS is of double
%   precision data type with complex values.
%
%   [Y,PATHGAINS] = step(CHAN,X,SELTX/SELRX) or step(CHAN,X,SELTX,SELRX)
%   returns the MIMO channel path gains for antenna selection schemes.
%   PATHGAINS is still of size Ns x Np x Nt x Nr with NaN values for the
%   unselected transmit-receive antenna pairs.
% 
%   [Y,PATHGAINS] = step(CHAN,X,...,INITIALTIME) returns the MIMO channel
%   path gains with the fading process starting at INITIALTIME. PATHGAINS
%   is still of size Ns x Np x Nt x Nr with NaN values for the unselected
%   transmit-receive antenna pairs.
% 
%   When the Visualization property is not set to 'Off', the selected
%   channel characteristics such as impulse response, frequency response or
%   Doppler spectrum are animated in separate figures, from the step
%   method.
% 
%   System objects may be called directly like a function instead of using
%   the step method. For example, y = step(obj, x) and y = obj(x) are
%   equivalent.
%
%   MIMOChannel2 methods:
%
%   step     - Filter input signal through a MIMO fading channel (see above)
%   release  - Allow property value and input characteristics changes
%   clone    - Create MIMO channel object with same property values
%   isLocked - Locked status (logical)
%   reset    - Reset states of filters, and random stream if the
%              RandomStream property is set to 'mt19937ar with seed'
%   info     - Return characteristic information about the MIMO channel
%
%   MIMOChannel2 properties:
%
%   SampleRate                      - Input signal sample rate (Hz)
%   PathDelays                      - Discrete path delay vector (s)
%   AveragePathGains                - Average path gain vector (dB)
%   NormalizePathGains              - Normalize path gains (logical)
%   FadingDistribution              - Rayleigh or Rician fading
%   KFactor                         - Rician K-factor scalar or vector (linear scale)
%   DirectPathDopplerShift          - Doppler shift(s) of line-of-sight component(s) (Hz)
%   DirectPathInitialPhase          - Initial phase(s) of line-of-sight component(s) (rad)
%   MaximumDopplerShift             - Maximum Doppler shift (Hz)
%   DopplerSpectrum                 - Doppler spectrum 
%   SpatialCorrelationSpecification - Specify spatial correlation
%   NumTransmitAntennas             - Number of transmit antennas
%   NumReceiveAntennas              - Number of receive antennas    
%   TransmitCorrelationMatrix       - Transmit correlation matrix (or 3-D array)
%   ReceiveCorrelationMatrix        - Receive correlation matrix (or 3-D array)
%   SpatialCorrelationMatrix        - Combined correlation matrix (or 3-D array)
%   AntennaSelection                - Optional transmit and/or receive antenna selection
%   NormalizeChannelOutputs         - Normalize channel outputs (logical)
%   FadingTechnique                 - Technique for generating fading samples
%   NumSinusoids                    - Number of sinusoids in sum-of-sinusoids technique
%   InitialTimeSource               - Initial time source for sum-of-sinusoids technique
%   InitialTime                     - Start time, in seconds, for sum-of-sinusoids technique
%   RandomStream                    - Source of random number stream
%   Seed                            - Initial seed of mt19937ar random number stream
%   PathGainsOutputPort             - Enable path gain output (logical)
%   Visualization                   - Optional channel visualization 
%   AntennaPairsToDisplay       - Transmit and receive antenna pair for visualization
%   PathsForDopplerDisplay      - Path for Doppler spectrum visualization
%   SamplesToDisplay            - Percentage of samples to be visualized
% 
%   % Example 1: 
%   %   Filter a 1000Hz input signal through a 2x2 Rayleigh
%   %   frequency-selective spatially correlated fading channel with a  
%   %   Jakes Doppler spectrum with a maximum frequency of 5Hz.
%
%   psk = comm.PSKModulator; 
%   modData = psk(randi([0 psk.ModulationOrder-1],1e5,1));
%   % Split modulated data into two spatial streams
%   channelInput = reshape(modData, [2, 5e4]).';
%   chan = comm.MIMOChannel2(...
%       'SampleRate',                1000,...
%       'PathDelays',                [0 1e-3],...
%       'AveragePathGains',          [3 5],...
%       'NormalizePathGains',        false,...
%       'MaximumDopplerShift',       5,...
%       'TransmitCorrelationMatrix', cat(3, eye(2), [1 0.1;0.1 1]),...
%       'ReceiveCorrelationMatrix',  cat(3, [1 0.2;0.2 1], eye(2)),...
%       'RandomStream',              'mt19937ar with seed',...
%       'Seed',                      33,...
%       'PathGainsOutputPort',       true);
%   [channelOutput, pathGains] = chan(channelInput);
%   % Check transmit and receive spatial correlation that should be close
%   % to the values of the TransmitCorrelationMatrix and
%   % ReceiveCorrelationMatrix properties of chan, respectively.
%   disp('Tx spatial correlation, first path, first Rx:');
%   disp(corrcoef(squeeze(pathGains(:,1,:,1)))); % Close to an identity matrix
%   disp('Tx spatial correlation, second path, second Rx:');
%   disp(corrcoef(squeeze(pathGains(:,2,:,2)))); % Close to [1 0.1;0.1 1]
%   disp('Rx spatial correlation, first path, second Tx:');
%   disp(corrcoef(squeeze(pathGains(:,1,2,:)))); % Close to [1 0.2;0.2 1]
%   disp('Rx spatial correlation, second path, first Tx:');
%   disp(corrcoef(squeeze(pathGains(:,2,1,:)))); % Close to an identity matrix
%
%   % Now enable transmit and receive antenna selection for the System 
%   % object chan. The input frame size is shortened to 100.
%   release(chan);
%   chan.AntennaSelection = 'Tx and Rx';
%   modData = psk(randi([0 psk.ModulationOrder-1],1e2,1)); 
%   % First transmit and second receive antennas are selected 
%   [channelOutput, pathGains] = chan(modData, [1 0], [0 1]); 
%   % Check the returned path gains have NaN values for those unselected 
%   % transmit-receive antenna pairs.
%   disp('Return 1 if the path gains for the second transmit antenna are NaN:');
%   disp(isequal(isnan(squeeze(pathGains(:,:,2,:))), ones(1e2, 2, 2)));
%   disp('Return 1 if the path gains for the first receive antenna are NaN:');
%   disp(isequal(isnan(squeeze(pathGains(:,:,:,1))), ones(1e2, 2, 2)));
%   
%   % Example 2: 
%   %   Filter an input signal using the 'Sum of sinusoids' technique for
%   %   the fading process. The input signal is first filtered through the
%   %   channel with 'InitialTimeSource' set to 'Property'. The same signal
%   %   is converted into frames and each frame is independently filtered 
%   %   through the same channel with 'InitialTimeSource' set to 
%   %   'Input port'. The fading samples of both implementations are
%   %   compared.
%
%   qpsk = comm.QPSKModulator('BitInput',true);
%   modData = qpsk(randi([0 1],2000,1)); 
%   % Split modulated data into two spatial streams
%   channelInput = reshape(modData, [2, 500]).';
%   chan = comm.MIMOChannel2(...
%       'SampleRate', 1000,...
%       'NormalizePathGains', false,...
%       'MaximumDopplerShift',10,... 
%       'PathGainsOutputPort', true, ...
%       'RandomStream', 'mt19937ar with seed', ...
%       'SpatialCorrelationSpecification', 'None', ...
%       'NumReceiveAntennas', 2,...
%       'NumTransmitAntennas', 2, ...
%       'FadingTechnique', 'Sum of sinusoids');
%    
%   %   Filter an input signal through a channel. The fading process starts
%   %   at InitialTime = 0.
%   [~,pathGains1] = chan(channelInput);
%    
%   %   The input signal is converted into frames and each frame is
%   %   independently filtered through the same channel. The frames are
%   %   transmitted consecutively. The transmission time of successive
%   %   frames is controlled by the initialTime value.
%   release(chan);
%   frameSpacing = 100;          % The spacing between frames in samples           
%   frameSize = 10;              % Frame size in samples   
%   pathGains2 = zeros(length(channelInput),1, ...
%               chan.NumTransmitAntennas,chan.NumReceiveAntennas);
%   chan.InitialTimeSource = 'Input port';
%   for i=1:(length(channelInput)/frameSpacing)
%       inIdx = frameSpacing*(i-1) + (1:frameSize);
%       initialTime = (inIdx(1)-1)* (1/chan.SampleRate);
%       [~, pathGains2(inIdx,1,:,:)] = chan(channelInput(inIdx,:), initialTime);  
%   end
%   %  Plot fading samples for transmit and receive antenna 1.
%   plot(abs(pathGains1(:,1,1,1)),'o-b'); hold on;
%   plot(abs(pathGains2(:,1,1,1)),'*-r'); grid on; axis square;
%   legend('InitialTimeSource : Property', 'InitialTimeSource : Input port');
%   xlabel('Time (s)'); ylabel('|Output|');
%
%   See also comm.AWGNChannel, comm.RayleighChannel, comm.RicianChannel,
%   doppler.

% Copyright 2011-2018 The MathWorks, Inc.

%#codegen
%#ok<*EMCA>

% Public properties
properties (Nontunable)
    %FadingDistribution Fading distribution
    %   Specify the fading distribution of the channel as one of 'Rayleigh'
    %   | 'Rician'. The default value of this property is 'Rayleigh', i.e.,
    %   the channel is Rayleigh fading.
    FadingDistribution = 'Rayleigh';
    %KFactor K-factors
    %   Specify the K factor of a Rician fading channel as a double
    %   precision, real, positive scalar or nonnegative, non-zero row
    %   vector of the same length as PathDelays. This property applies when
    %   you set the FadingDistribution property to 'Rician'. If KFactor is
    %   a scalar, the first discrete path is a Rician fading process with a
    %   Rician K-factor of KFactor and the remaining discrete paths are
    %   independent Rayleigh fading processes. If KFactor is a row vector,
    %   the discrete path corresponding to a positive element of the
    %   KFactor vector is a Rician fading process with a Rician K-factor
    %   specified by that element and the discrete path corresponding to a
    %   zero-valued element of the KFactor vector is a Rayleigh fading
    %   process. The default value of this property is 3.
    KFactor = 3;
    
    mvalue = 5;
    %DirectPathDopplerShift LOS path Doppler shifts (Hz)
    %   Specify the Doppler shift(s) of the line-of-sight component(s) of a
    %   Rician fading channel in Hz as a double precision, real scalar or
    %   row vector. This property applies when you set the
    %   FadingDistribution property to 'Rician'. DirectPathDopplerShift
    %   must have the same size as KFactor. If DirectPathDopplerShift is a
    %   scalar, it is the line-of-sight component Doppler shift of the
    %   first discrete path that is a Rician fading process. If
    %   DirectPathDopplerShift is a row vector, the discrete path that is a
    %   Rician fading process indicated by a positive element of the
    %   KFactor vector has its line-of-sight component Doppler shift
    %   specified by the corresponding element of DirectPathDopplerShift.
    %   The default value of this property is 0.
    DirectPathDopplerShift = 0;
    %DirectPathInitialPhase LOS path initial phases (rad)
    %   Specify the initial phase(s) of the line-of-sight component(s) of a
    %   Rician fading channel in radians as a double precision, real scalar
    %   or row vector. This property applies when you set the
    %   FadingDistribution property to 'Rician'. DirectPathInitialPhase
    %   must have the same size as KFactor. If DirectPathInitialPhase is a
    %   scalar, it is the line-of-sight component initial phase of the
    %   first discrete path that is a Rician fading process. If
    %   DirectPathInitialPhase is a row vector, the discrete path that is a
    %   Rician fading process indicated by a positive element of the
    %   KFactor vector has its line-of-sight component initial phase
    %   specified by the corresponding element of DirectPathInitialPhase.
    %   The default value of this property is 0.
    DirectPathInitialPhase = 0;
    % DopplerSpectrum Doppler spectrum
    %   Specify the Doppler spectrum shape for the path(s) of the channel.
    %   This property accepts a single Doppler spectrum structure returned
    %   from the doppler function or a row cell array of such structures.
    %   The maximum Doppler shift value necessary to specify the Doppler
    %   spectrum/spectra is given by the MaximumDopplerShift property. This
    %   property applies when the MaximumDopplerShift property value is
    %   greater than 0.
    %   
    %   If you assign a single Doppler spectrum structure to
    %   DopplerSpectrum, all paths have the same specified Doppler
    %   spectrum. The possible Doppler spectrum structures are
    %       doppler('Jakes')
    %       doppler('Flat')
    %       doppler('Rounded', ...)
    %       doppler('Bell', ...)
    %       doppler('Asymmetric Jakes', ...)
    %       doppler('Restricted Jakes', ...)
    %       doppler('Gaussian', ...) 
    %       doppler('BiGaussian', ...)
    %   
    %   If you assign a row cell array of different Doppler spectrum
    %   structures (which can be chosen from any of those listed above) to
    %   DopplerSpectrum, each path has the Doppler spectrum specified by
    %   the corresponding structure in the cell array. In this case, the
    %   length of DopplerSpectrum must be equal to the length of
    %   PathDelays.
    % 
    %   Alternatively, you can specify DopplerSpectrum as a single Doppler
    %   spectrum object or a row vector of such objects that must have a
    %   length equal to the length of PathDelays. The possible Doppler
    %   spectrum objects are
    %       doppler.jakes
    %       doppler.flat
    %       doppler.rounded(...)
    %       doppler.bell(...)
    %       doppler.ajakes(...)
    %       doppler.rjakes(...)
    %       doppler.gaussian(...)
    %       doppler.bigaussian(...)
    %
    %   To generate code, specify this property to a single Doppler
    %   spectrum structure. The default value of this property is
    %   doppler('Jakes').
    DopplerSpectrum = doppler('Jakes'); 
    %SpatialCorrelationSpecification Specify spatial correlation
    %   Specify spatial correlation as one of 'None' | 'Separate Tx Rx' |
    %   'Combined', where Tx means transmit antennas and Rx means receive
    %   antennas. Set this property to 'None' to specify the number of
    %   transmit and receive antennas. Set this property to 'Spatial Tx Rx'
    %   to separately specify the transmit and receive spatial correlation
    %   matrices from which the number of transmit antenna (Nt) and number
    %   of receive antennas (Nr) are derived. Set this property to
    %   'Combined' to specify the single correlation matrix for the whole
    %   channel, from which the product of Nt and Nr is derived. The
    %   default value of this property is 'Separate Tx Rx'.
    SpatialCorrelationSpecification = 'Separate Tx Rx'    
    %NumTransmitAntennas Number of transmit antennas
    %   Specify the number of transmit antennas as a numeric, positive
    %   integer scalar. This property applies when you set the
    %   SpatialCorrelationSpecification property to 'None' or 'Combined'.
    %   The default value of this property is 2.
    NumTransmitAntennas = 2;
    %NumReceiveAntennas Number of receive antennas
    %   Specify the number of receive antennas as a numeric, positive
    %   integer scalar. This property applies when you set the
    %   SpatialCorrelationSpecification property to 'None'. The default
    %   value of this property is 2.
    NumReceiveAntennas = 2;
    %TransmitCorrelationMatrix Transmit spatial correlation
    %   Specify the spatial correlation of the transmitter as a double
    %   precision, 2D matrix or 3D array. This property applies when you
    %   set the SpatialCorrelationSpecification property to 'Separate Tx
    %   Rx'. The first dimension of TransmitCorrelationMatrix determines
    %   the number of transmit antennas Nt. If the channel is
    %   frequency-flat, i.e., PathDelays is a scalar,
    %   TransmitCorrelationMatrix is a 2D Hermitian matrix of size Nt x Nt.
    %   The magnitude of any off-diagonal element must be no larger than
    %   the geometric mean of the two corresponding diagonal elements.
    %  
    %   If the channel is frequency-selective, i.e., PathDelays is a row
    %   vector of length Np, TransmitCorrelationMatrix can be specified as
    %   a 2D matrix, in which case each path has the same transmit spatial
    %   correlation matrix. Alternatively, it can be specified as a 3-D
    %   array of size Nt x Nt x Np, in which case each path can have its
    %   own different transmit spatial correlation matrix.
    % 
    %   The default value of this property is [1 0;0 1].
    TransmitCorrelationMatrix = eye(2);
    %ReceiveCorrelationMatrix Receive spatial correlation
    %   Specify the spatial correlation of the receiver as a double
    %   precision, 2D matrix or 3D array. This property applies when you
    %   set the SpatialCorrelationSpecification property to 'Separate Tx
    %   Rx'. The first dimension of ReceiveCorrelationMatrix determines the
    %   number of receive antennas Nr. If the channel is frequency-flat,
    %   i.e., PathDelays is a scalar, ReceiveCorrelationMatrix is a 2D
    %   Hermitian matrix of size Nr x Nr. The magnitude of any off-diagonal
    %   element must be no larger than the geometric mean of the two
    %   corresponding diagonal elements.
    %  
    %   If the channel is frequency-selective, i.e., PathDelays is a row
    %   vector of length Np, ReceiveCorrelationMatrix can be specified as
    %   a 2D matrix, in which case each path has the same receive spatial
    %   correlation matrix. Alternatively, it can be specified as a 3-D
    %   array of size Nr x Nr x Np, in which case each path can have its
    %   own different receive spatial correlation matrix.
    % 
    %   The default value of this property is [1 0;0 1].
    ReceiveCorrelationMatrix = eye(2);
    %SpatialCorrelationMatrix Combined spatial correlation
    %   Specify the combined spatial correlation for the channel as a
    %   double precision, 2D matrix or 3D array. This property applies when
    %   you set the SpatialCorrelationSpecification property to 'Combined'.
    %   The first dimension of SpatialCorrelationMatrix determines the
    %   product of the number of transmit antennas Nt and the number of
    %   receive antennas Nr. If the channel is frequency-flat, i.e.,
    %   PathDelays is a scalar, SpatialCorrelationMatrix is a 2D Hermitian
    %   matrix of size (NtNr) x (NtNr). The magnitude of any off-diagonal
    %   element must be no larger than the geometric mean of the two
    %   corresponding diagonal elements.
    %  
    %   If the channel is frequency-selective, i.e., PathDelays is a row
    %   vector of length Np, SpatialCorrelationMatrix can be specified as a
    %   2D matrix, in which case each path has the same spatial correlation
    %   matrix. Alternatively, it can be specified as a 3-D array of size
    %   (NtNr) x (NtNr) x Np, in which case each path can have its own
    %   different spatial correlation matrix.
    % 
    %   The default value of this property is [1 0 0 0; 0 1 0 0; 0 0 1 0; 
    %   0 0 0 1].
    SpatialCorrelationMatrix = eye(4);
    %AntennaSelection Antenna selection
    %   Specify the antenna selection scheme as one of 'Off' | 'Tx' | 'Rx'
    %   | 'Tx and Rx', where Tx means transmit antennas and Rx means
    %   receive antennas. When antenna selection is on at Tx and/or Rx,
    %   additional input(s) is required to specify which antennas are
    %   selected for signal transmission. The default value of this
    %   property is 'Off'.
    AntennaSelection = 'Off';
    %AntennaPairsToDisplay Antenna pair to display 
    %   Specify a pair of transmit and receive antennas to be visualized as
    %   a numeric, real, 1 x 2 integer row vector in [1, Nt] x [1, Nr],
    %   where Nt and Nr are the number of transmit and receive antennas
    %   determined by NumTransmitAntennas and NumReceiveAntennas or
    %   TransmitCorrelationMatrix and ReceiveCorrelationMatrix
    %   respectively. This property applies when you set the Visualization
    %   property to 'Impulse response', 'Frequency response', 'Impulse and
    %   frequency responses' or 'Doppler spectrum'. The default value of
    %   this property is [1 1].
    AntennaPairsToDisplay = [1 1];    
end

properties (Nontunable, Logical)
    %NormalizeChannelOutputs Normalize outputs by number of receive antennas
    %   Set this property to true to normalize the channel outputs by the
    %   number of receive antennas. The default value of this property is
    %   true.
    NormalizeChannelOutputs = true;
end

properties (Nontunable, Dependent)
    %SpatialCorrelation Spatially correlated antennas
    %   Set this property to true to specify the transmit and receive
    %   spatial correlation matrices from which the number of transmit and
    %   receive antennas can be derived. Set this property to false to
    %   specify the number of transmit and receive antennas instead. In
    %   this case, the transmit and receive spatial correlation matrices
    %   are both identity matrices. The default value of this property is
    %   true.
    SpatialCorrelation = true; 
end

properties(Constant, Hidden)
    RandomNumGenerator = 'mt19937ar'; 
    FadingDistributionSet = matlab.system.StringSet({ ...
        'Rayleigh','nakagami'});
    SpatialCorrelationSpecificationSet = matlab.system.StringSet({ ...
        'None','Separate Tx Rx','Combined'});    
    AntennaSelectionSet = matlab.system.StringSet({ ...
        'Off','Tx','Rx','Tx and Rx'});    
end

methods
  function obj = MIMOChannel2(varargin) % Constructor
    setProperties(obj, nargin, varargin{:});
  end 
  
  function set.KFactor(obj, K)
    propName = 'KFactor'; 
    validateattributes(K, {'double'}, ...         
         {'real','row','finite','nonnegative'}, ...
         [class(obj) '.' propName], propName); 
    
    coder.internal.errorIf(all(K == 0), 'comm:FadingChannel:KFactorAllZero');
    
    obj.KFactor = K;
  end
  
  function set.mvalue(obj, m)
    propName = 'mvalue'; 
    validateattributes(m, {'double'}, ...         
         {'real','row','finite','nonnegative'}, ...
         [class(obj) '.' propName], propName); 
    
    coder.internal.errorIf(all(m < 0.5), 'comm:FadingChannel:KFactorAllZero');
    
    obj.mvalue = m;
  end
        
  function set.DirectPathDopplerShift(obj, LOSShift)
    propName = 'DirectPathDopplerShift';
    validateattributes(LOSShift, {'double'}, {'real','row','finite'}, ...
        [class(obj) '.' propName], propName); 

    obj.DirectPathDopplerShift = LOSShift;
  end
    
  function set.DirectPathInitialPhase(obj, LOSPhase)
    propName = 'DirectPathInitialPhase';
    validateattributes(LOSPhase, {'double'}, {'real','row','finite'}, ...
        [class(obj) '.' propName], propName); 

    obj.DirectPathInitialPhase = LOSPhase;
  end
  
  function set.DopplerSpectrum(obj, ds)
    coder.extrinsic('num2str', ...
        'CustomChannelBase.getCutoffFreqFactor');
    
    propName = 'DopplerSpectrum';
    validateattributes(ds, {'doppler.baseclass','struct','cell'}, {'row'}, ... 
         [class(obj) '.' propName], propName);     
    
    if isa(ds, 'cell') 
        for idx = 1:length(ds)
            validateattributes(ds{idx}, {'struct'}, {}, ... 
                 [class(obj) '.' propName], [propName, '{', num2str(idx), '}']);
        end
    end
    
    % Set up cutoff frequency factor (not equal to 0) for each discrete
    % path. If ds is a structure or a cell array of structures, it is also
    % validated in the extrinsic function. 
    if isempty(coder.target)
        obj.pFcFactor = CustomChannelBase.getCutoffFreqFactor(ds); 
    else
        obj.pFcFactor = coder.const(double( ...
            CustomChannelBase.getCutoffFreqFactor(ds))); 
    end
    
    if isa(ds, 'doppler.baseclass') 
        obj.DopplerSpectrum = copy(ds);
    else 
        obj.DopplerSpectrum = ds;
    end
  end
  
  function set.SpatialCorrelation(obj, corrFlag)
    propName = 'SpatialCorrelation';
    validateattributes(corrFlag, {'logical'}, {'scalar'}, ...
        [class(obj) '.' propName], propName); 

    if corrFlag
        obj.SpatialCorrelationSpecification = 'Separate Tx Rx';
    else
        obj.SpatialCorrelationSpecification = 'None';
    end      
  end
  
  function flag = get.SpatialCorrelation(obj)
     flag = ~strcmp(obj.SpatialCorrelationSpecification, 'None');
  end  
  
  function set.NumTransmitAntennas(obj, Nt)
    propName = 'NumTransmitAntennas';
    validateattributes(Nt, {'numeric'}, ... 
        {'real','scalar','integer','positive'}, ... 
        [class(obj) '.' propName], propName); 

    obj.NumTransmitAntennas = Nt;
  end   
  
  function set.NumReceiveAntennas(obj, Nr)
    propName = 'NumReceiveAntennas';
    validateattributes(Nr, {'numeric'}, ...
        {'real','scalar','integer','positive'}, ...
        [class(obj) '.' propName], propName); 

    obj.NumReceiveAntennas = Nr;  
  end
  
  function set.TransmitCorrelationMatrix(obj, Rt)
    propName = 'TransmitCorrelationMatrix';
    validateattributes(Rt, {'double'}, {'finite','nonempty'}, ...
        [class(obj) '.' propName], propName); 

    coder.internal.errorIf(ndims(Rt) > 3, ...
        'comm:MIMOChannel2:CorrMtxMoreThan3D', 'TransmitCorrelationMatrix');
    
    for i = 1:size(Rt,3)
        validateCorrelationMatrix(Rt(:,:,i));
    end

    obj.TransmitCorrelationMatrix = Rt;
  end
 
  function set.ReceiveCorrelationMatrix(obj, Rr)
    propName = 'ReceiveCorrelationMatrix';
    validateattributes(Rr, {'double'}, {'finite','nonempty'}, ...
        [class(obj) '.' propName], propName); 
    
    coder.internal.errorIf(ndims(Rr) > 3, ... 
        'comm:MIMOChannel2:CorrMtxMoreThan3D', 'ReceiveCorrelationMatrix');
    
    for i = 1:size(Rr,3)
        validateCorrelationMatrix(Rr(:,:,i));
    end
           
    obj.ReceiveCorrelationMatrix = Rr;
  end 
  
  function set.SpatialCorrelationMatrix(obj, Rspat)
    propName = 'SpatialCorrelationMatrix';
    validateattributes(Rspat, {'double'}, {'finite','nonempty'}, ...
        [class(obj) '.' propName], propName); 

    coder.internal.errorIf(ndims(Rspat) > 3, ...
        'comm:MIMOChannel2:CorrMtxMoreThan3D', 'SpatialCorrelationMatrix');
    
    for i = 1:size(Rspat,3)
        validateCorrelationMatrix(Rspat(:,:,i));
    end

    obj.SpatialCorrelationMatrix = Rspat;
  end
  
  function set.NormalizeChannelOutputs(obj, v)
    propName = 'NormalizeChannelOutputs';
    validateattributes(v, {'logical'}, {'scalar'}, ...
        [class(obj) '.' propName], propName); 

    obj.NormalizeChannelOutputs = v;
  end    
  
  function set.AntennaPairsToDisplay(obj, dispLink)
    propName = 'AntennaPairsToDisplay';
    validateattributes(dispLink, {'numeric'}, ...
        {'real','integer','>=',1,'size',[1 2]}, ...
        [class(obj) '.' propName], propName); 

    obj.AntennaPairsToDisplay = dispLink;
  end  
end

methods(Access = protected)
  function validatePropertiesImpl(obj)    
    % Check KFactor, DirectPathDopplerShift, DirectPathInitialPhase sizes
    if strcmp(obj.FadingDistribution, 'nakagami')
        validateNakagamiProperties(obj);
    end
        
    % Check 3rd dimension of correlation matrix match number of paths
    if ~strcmp(obj.SpatialCorrelationSpecification, 'None')
        corrMtxProp = {'TransmitCorrelationMatrix', ...
            'ReceiveCorrelationMatrix', 'SpatialCorrelationMatrix'};
        NP = length(obj.PathDelays);
        
        for i = 1:length(corrMtxProp)
            if ~isInactivePropertyImpl(obj, corrMtxProp{i})
                R = obj.(corrMtxProp{i});
                % We allow ones(1,1,NP) when Nt = 1 and/or Nr = 1. 
                coder.internal.errorIf(...
                    (ndims(R) == 3) && (size(R, 3) ~= NP), ...
                    'comm:MIMOChannel2:CorrMtxDimNotMatchNP', corrMtxProp{i});
            end
        end
        
        coder.internal.errorIf( ...
            strcmp(obj.SpatialCorrelationSpecification, 'Combined') && ...
            mod(size(obj.SpatialCorrelationMatrix, 1), ...
            obj.NumTransmitAntennas) ~= 0, 'comm:MIMOChannel2:NtNotDivCorrMtxDim');
    end

    % Check # of Tx/Rx when antenna selection is on
    [Nt, Nr] = getNumTxAndRx(obj);
    coder.internal.errorIf((Nt == 1) && ...
       any(strcmp(obj.AntennaSelection, {'Tx','Tx and Rx'})), ...
       'comm:MIMOChannel2:SelectionOnFor1Tx'); 

    coder.internal.errorIf((Nr == 1) && ...
       any(strcmp(obj.AntennaSelection, {'Rx','Tx and Rx'})), ...
       'comm:MIMOChannel2:SelectionOnFor1Rx');    
    
    % Check AntennaPairsToDisplay value
    coder.internal.errorIf(~strcmp(obj.Visualization, 'Off') && ...
        any(obj.AntennaPairsToDisplay > [Nt, Nr]), ...
        'comm:MIMOChannel2:DispAntPairNotInNtxNr');
    
    % Call shared property validation 
    validatePropertiesImpl@CustomChannelBase(obj);    
  end
  
  function flag = isInactivePropertyImpl(obj, prop)
    % Use the if-else format for codegen
    if strcmp(prop, 'NumTransmitAntennas') 
        flag = strcmp(obj.SpatialCorrelationSpecification, 'Separate Tx Rx'); 
    elseif strcmp(prop, 'NumReceiveAntennas') 
        flag = ~strcmp(obj.SpatialCorrelationSpecification, 'None'); 
    elseif any(strcmp(prop, {'TransmitCorrelationMatrix', 'ReceiveCorrelationMatrix'}))
        flag = ~strcmp(obj.SpatialCorrelationSpecification, 'Separate Tx Rx'); 
    elseif strcmp(prop, 'SpatialCorrelationMatrix') 
        flag = ~strcmp(obj.SpatialCorrelationSpecification, 'Combined'); 
    elseif any(strcmp(prop, {'KFactor', 'DirectPathDopplerShift', 'DirectPathInitialPhase'}))
        flag = strcmp(obj.FadingDistribution, 'Rayleigh');
    elseif strcmp(prop, 'AntennaPairsToDisplay')
        flag = strcmp(obj.Visualization, 'Off');
    else
        flag = isInactivePropertyImpl@comm.internal.FadingChannel(obj, prop);
    end
  end  
end

methods(Static, Hidden)
  function blk = getAlternateBlock
    blk = 'commchan3/MIMO Fading Channel';
  end    
end

methods(Static, Access = protected)    
  function groups = getPropertyGroupsImpl    
    multipath = matlab.system.display.Section( ...
        'Title', 'Multipath parameters (frequency selectivity)', ...
        'PropertyList', {'SampleRate', 'PathDelays', 'AveragePathGains', ...
        'NormalizePathGains', 'FadingDistribution', 'KFactor', ...
        'DirectPathDopplerShift', 'DirectPathInitialPhase'});
        
    doppler = matlab.system.display.Section( ...
        'Title', 'Doppler parameters (time dispersion)', ...
        'PropertyList', {'MaximumDopplerShift', 'DopplerSpectrum'});
    
    antenna = matlab.system.display.Section(...
        'Title', 'Antenna parameters (spatial dispersion)', ...
        'PropertyList', {'SpatialCorrelationSpecification', ...
        'SpatialCorrelationMatrix', 'NumTransmitAntennas', ...
        'NumReceiveAntennas', 'TransmitCorrelationMatrix', ...
        'ReceiveCorrelationMatrix', 'AntennaSelection', ...
        'NormalizeChannelOutputs'}); 
    
    pRandStream = matlab.system.display.internal.Property('RandomStream', ...
        'IsGraphical', false, ...
        'UseClassDefault', false, ...
        'Default', 'mt19937ar with seed');
    
    randomization = matlab.system.display.Section(...
        'PropertyList', {pRandStream, 'Seed','PathGainsOutputPort'});
    
    mainGroup = matlab.system.display.SectionGroup(...
        'TitleSource', 'Auto', ...
        'Sections', [multipath doppler antenna]);
    
    fadingTechnique =  matlab.system.display.Section(...
        'PropertyList',{'FadingTechnique','NumSinusoids','InitialTimeSource', ...
        'InitialTime'});

    visual = matlab.system.display.Section(...
        'PropertyList', {'Visualization', 'AntennaPairsToDisplay', ...
        'PathsForDopplerDisplay', 'SamplesToDisplay'});    
    
    realizationGroup = matlab.system.display.SectionGroup(...
        'Title', 'Realization', ...
        'Sections', [fadingTechnique randomization]);
    
    visualGroup = matlab.system.display.SectionGroup(...
        'Title', 'Visualization', ...
        'Sections', visual);    

    groups = [mainGroup realizationGroup visualGroup];
  end
end
end
%==========================================================================
% Support functions
%==========================================================================
function validateCorrelationMatrix(M)

% Check Hermitian
coder.internal.errorIf((size(M, 1) ~= size(M, 2)) || ...
    ~all(all(abs(M-M') <= sqrt(eps))), 'comm:MIMOChannel2:CorrMtxNotHermitian');

% Check diagonal elements
coder.internal.errorIf(any(diag(M) <= 0), 'comm:MIMOChannel2:CorrMtxNonPosDiag');

% Check off-diagonal elements
M2 = M.*conj(M);

for i = 1:size(M,1)
    M2(i+1:end,i) = M2(i+1:end,i)/M(i,i);
    M2(i,1:i-1) = M2(i,1:i-1)/M(i,i);
    M2(i,i) = 1;
end

coder.internal.errorIf(~all(all(real(tril(M2)) <= 1)), ...
    'comm:MIMOChannel2:CorrMtxViolatesSchwartz');

end

% [EOF]
