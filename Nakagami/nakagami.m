classdef (StrictDefaults)nakagami < CustomChannelBase
%RicianChannel Filter input signal through a Rician fading channel
%   CHAN = comm.RicianChannel creates a frequency-selective or
%   frequency-flat multipath Rician fading channel System object, CHAN.
%   This object filters a real or complex input signal through the
%   multipath channel to obtain the channel impaired signal.
%
%   CHAN = comm.RicianChannel(Name,Value) creates a multipath Rician fading
%   channel object, CHAN, with the specified property Name set to the
%   specified Value. You can specify additional name-value pair arguments
%   in any order as (Name1,Value1,...,NameN,ValueN).
%
%   Step method syntax:
%
%   Y = step(CHAN,X) filters input signal X through a multipath Rician 
%   fading channel and returns the result in Y. Both the input X and the
%   output signal Y are of size Ns x 1, where Ns is the number of samples.
%   The input X can be of double precision data type with real or complex
%   values. Y is of double precision data type with complex values.
%
%   [Y,PATHGAINS] = step(CHAN,X) returns the channel path gains of the
%   underlying Rician fading process in PATHGAINS. This syntax applies when
%   you set the PathGainsOutputPort property of CHAN to true. PATHGAINS is
%   of size Ns x Np, where Np is the number of paths, i.e., the length of
%   the PathDelays property value of CHAN. PATHGAINS is of double precision
%   data type with complex values.
%
%   Y = step(CHAN,X,INITIALTIME) filters input signal X through a multipath
%   Rician fading channel, with the fading process starting at time
%   INITIALTIME and returns the result in Y. This syntax applies when you
%   set the 'FadingTechnique' property of H to 'Sum of sinusoids' and the
%   'InitialTimeSource' property of H to 'Input port'.
%
%   [Y,PATHGAINS] = step(CHAN,X,INITIALTIME) returns the channel path gains
%   of the Rician fading process in PATHGAINS, with the fading process
%   starting at time INITIALTIME. This syntax applies when you set the
%   'FadingTechnique' property of CHAN to 'Sum of sinusoids' and the
%   'InitialTimeSource' property of CHAN to 'Input port'.
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
%   RicianChannel methods:
%
%   step     - Filter input signal through a Rician fading channel (see above)
%   release  - Allow property value and input characteristics changes
%   clone    - Create Rician channel object with same property values
%   isLocked - Locked status (logical)
%   reset    - Reset states of filters, and random stream if the
%              RandomStream property is set to 'mt19937ar with seed'
%   info     - Return characteristic information about the Rician channel
%
%   RicianChannel properties:
%
%   SampleRate              - Input signal sample rate (Hz)
%   PathDelays              - Discrete path delay vector (s)
%   AveragePathGains        - Average path gain vector (dB)
%   NormalizePathGains      - Normalize path gains (logical)
%   KFactor                 - Rician K-factor scalar or vector (linear scale)
%   mvalue                  - Nakagami m-value for power constraints
%   DirectPathDopplerShift  - Doppler shift(s) of line-of-sight component(s) (Hz)
%   DirectPathInitialPhase  - Initial phase(s) of line-of-sight component(s) (rad)
%   MaximumDopplerShift     - Maximum Doppler shift (Hz)
%   DopplerSpectrum         - Doppler spectrum
%   FadingTechnique         - Technique for generating fading samples
%   NumSinusoids            - Number of sinusoids in sum-of-sinusoids technique
%   InitialTimeSource       - Initial time source for sum-of-sinusoids technique
%   InitialTime             - Start time, in seconds, for sum-of-sinusoids technique
%   RandomStream            - Source of random number stream
%   Seed                    - Initial seed of mt19937ar random number stream
%   PathGainsOutputPort     - Enable path gain output (logical)
%   Visualization           - Optional channel visualization 
%   PathsForDopplerDisplay  - Path for Doppler spectrum visualization
%   SamplesToDisplay        - Percentage of samples to be visualized
%
%   % Example 1: 
%   %   How to produce repeatable outputs when a comm.RicianChannel System
%   %   object uses the global stream for random number generation.
%   
%   psk = comm.PSKModulator;
%   channelInput = psk(randi([0 psk.ModulationOrder-1],512,1));
%   chan = comm.RicianChannel(...
%       'SampleRate',             1e6,...
%       'PathDelays',             [0.0 0.5 1.2]*1e-6,...
%       'AveragePathGains',       [0.1 0.5 0.2],...
%       'KFactor',                2.8,...
%       'DirectPathDopplerShift', 5.0,...
%       'DirectPathInitialPhase', 0.5,...
%       'MaximumDopplerShift',    50,...
%       'DopplerSpectrum',        doppler('Bell', 8),...
%       'PathGainsOutputPort',    true);
% 
%   % Log current global stream state
%   globalStream = RandStream.getGlobalStream; 
%   loggedState  = globalStream.State;
%   
%   % Filter the modulated data for the first time
%   [RicianChanOut1, RicianPathGains1] = chan(channelInput);
%
%   % Set global stream back to the logged state and reset the channel
%   globalStream.State = loggedState;
%   reset(chan);
%
%   % Filter the modulated data for the second time
%   [RicianChanOut2, RicianPathGains2] = chan(channelInput);
%  
%   % Verify channel and path gain outputs are the same for two channel runs
%   display(isequal(RicianChanOut1, RicianChanOut2));
%   display(isequal(RicianPathGains1, RicianPathGains2));
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
%   qpskmod = comm.QPSKModulator('BitInput',true);
%   modData = qpskmod(randi([0 1],1000,1));   
%   chan = comm.RicianChannel(...
%       'SampleRate',             1e3,...
%       'KFactor',                2.8,...
%       'DirectPathDopplerShift', 10.0,...
%       'DirectPathInitialPhase', 0.5,...
%       'MaximumDopplerShift',    10,...
%       'PathGainsOutputPort',    true,...
%       'FadingTechnique',        'Sum of sinusoids',...
%       'InitialTimeSource',      'Property', ...
%       'RandomStream',           'mt19937ar with seed',...
%       'Seed',                   73);
%
%   % Filter an input signal through a channel. The fading process starts
%   % at InitialTime = 0.
%   [~,pathGains1] = chan(modData);
%    
%   % The input signal is converted into frames and each frame is
%   % independently filtered through the same channel. The frames are
%   % transmitted consecutively. The transmission time of successive
%   % frames is controlled by the initialTime.
%   release(chan);
%   frameSpacing = 100;      % The spacing between frames in samples 
%   frameSize = 10;          % Frame size in samples
%   pathGains2 = zeros(length(pathGains1),1);
%   chan.InitialTimeSource = 'Input port';
%   for i=1:(length(modData)/frameSpacing)
%      inIdx = frameSpacing*(i-1) + (1:frameSize);
%      initialTime = (inIdx(1)-1)* (1/chan.SampleRate);
%      [~, pathGains2(inIdx,:)] = chan(modData(inIdx,:), initialTime);  
%   end
%   % Plot fading samples
%   plot(abs(pathGains1),'o-b'); hold on;
%   plot(abs(pathGains2),'*-r'); grid on; axis square;
%   legend('InitialTimeSource : Property', 'InitialTimeSource : Input port');
%   xlabel('Time (s)'); ylabel('|Output|');
%   
%   See also comm.AWGNChannel, comm.RayleighChannel, comm.MIMOChannel,
%   doppler.

% Copyright 2012-2018 The MathWorks, Inc.

%#codegen

properties (Nontunable)
    %KFactor K-factors
    %   Specify the K factor of a Rician fading channel as a double
    %   precision, real, positive scalar or nonnegative, non-zero row
    %   vector of the same length as PathDelays. If KFactor is a scalar,
    %   the first discrete path is a Rician fading process with a Rician
    %   K-factor of KFactor and the remaining discrete paths are
    %   independent Rayleigh fading processes. If KFactor is a row vector,
    %   the discrete path corresponding to a positive element of the
    %   KFactor vector is a Rician fading process with a Rician K-factor
    %   specified by that element and the discrete path corresponding to a
    %   zero-valued element of the KFactor vector is a Rayleigh fading
    %   process. The default value of this property is 3.
    KFactor = 3;
    
    mvalue = 1; % mvalue >= 0.5
    %DirectPathDopplerShift LOS path Doppler shifts (Hz)
    %   Specify the Doppler shift(s) of the line-of-sight component(s) of a
    %   Rician fading channel in Hz as a double precision, real scalar or
    %   row vector. DirectPathDopplerShift must have the same size as
    %   KFactor. If DirectPathDopplerShift is a scalar, it is the
    %   line-of-sight component Doppler shift of the first discrete path
    %   that is a Rician fading process. If DirectPathDopplerShift is a row
    %   vector, the discrete path that is a Rician fading process indicated
    %   by a positive element of the KFactor vector has its line-of-sight
    %   component Doppler shift specified by the corresponding element of
    %   DirectPathDopplerShift. The default value of this property is 0.
    DirectPathDopplerShift = 0;
    %DirectPathInitialPhase LOS path initial phases (rad)
    %   Specify the initial phase(s) of the line-of-sight component(s) of a
    %   Rician fading channel in radians as a double precision, real scalar
    %   or row vector. DirectPathInitialPhase must have the same size as
    %   KFactor. If DirectPathInitialPhase is a scalar, it is the
    %   line-of-sight component initial phase of the first discrete path
    %   that is a Rician fading process. If DirectPathInitialPhase is a row
    %   vector, the discrete path that is a Rician fading process indicated
    %   by a positive element of the KFactor vector has its line-of-sight
    %   component initial phase specified by the corresponding element of
    %   DirectPathInitialPhase. The default value of this property is 0.
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
    %   To generate code, specify this property to a single Doppler
    %   spectrum structure. The default value of this property is
    %   doppler('Jakes').
    DopplerSpectrum = doppler('Jakes');
end

properties(Constant, Hidden)    
    FadingDistribution              = 'nakagami';
    SpatialCorrelationSpecification = 'None';
    NumTransmitAntennas             = 1;
    NumReceiveAntennas              = 1;
    AntennaSelection                = 'Off';
    NormalizeChannelOutputs         = false;
    AntennaPairsToDisplay           = [1 1];
    RandomNumGenerator              = 'mt19937ar'; 
end

methods
  function obj = RicianChannel(varargin) % Constructor
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
        'comm.internal.CustomChannelBase.getCutoffFreqFactor');
    
    propName = 'DopplerSpectrum';
    validateattributes(ds, {'struct','cell'}, {'row'}, ... 
         [class(obj) '.' propName], propName);     
    
    if isa(ds, 'cell') 
        for idx = 1:length(ds)
            validateattributes(ds{idx}, {'struct'}, {}, ... 
                 [class(obj) '.' propName], [propName, '{', num2str(idx), '}']);
        end
    end
    
    % Set up cutoff frequency factor (not equal to 0) for each discrete
    % path. The structure field(s) is validated in the extrinsic function. 
    if isempty(coder.target)
        obj.pFcFactor = comm.internal.CustomChannelBase.getCutoffFreqFactor(ds); 
    else
        obj.pFcFactor = coder.internal.const(double( ...
            comm.internal.CustomChannelBase.getCutoffFreqFactor(ds))); 
    end
    
    obj.DopplerSpectrum = ds;
  end
end

methods(Access = protected)   
  function validatePropertiesImpl(obj)    
    % Check KFactor, DirectPathDopplerShift, DirectPathInitialPhase sizes
    validateNakagamiProperties(obj);    
    
    % Call shared property validation 
    validatePropertiesImpl@CustomChannelBase(obj);
  end  
end

methods(Static, Hidden)
  function blk = getAlternateBlock
    blk = 'commchan3/SISO Fading Channel';
  end    
end

methods(Static, Access = protected)
  function groups = getPropertyGroupsImpl 
    multipath = matlab.system.display.Section( ...
        'Title', 'Multipath parameters (frequency selectivity)', ...
        'PropertyList', {'SampleRate', 'PathDelays', 'AveragePathGains', ...
        'NormalizePathGains', 'KFactor', 'DirectPathDopplerShift', ...
        'DirectPathInitialPhase'});
        
    doppler = matlab.system.display.Section( ...
        'Title', 'Doppler parameters (time dispersion)', ...
        'PropertyList', {'MaximumDopplerShift', 'DopplerSpectrum'});
        
    randomization = matlab.system.display.Section(...
        'PropertyList', {'RandomStream', 'Seed', 'PathGainsOutputPort'});
        
    mainGroup = matlab.system.display.SectionGroup(...
        'TitleSource', 'Auto', ...
        'Sections', [multipath doppler]);
    
    fadingTechnique =  matlab.system.display.Section(...
        'PropertyList',{'FadingTechnique','NumSinusoids','InitialTimeSource', ...
        'InitialTime'});
    
    realizationGroup = matlab.system.display.SectionGroup(...
        'Title', 'Realization', ...
        'Sections', [fadingTechnique randomization]);
    
    visual = matlab.system.display.Section(...
        'PropertyList', {'Visualization', 'PathsForDopplerDisplay', 'SamplesToDisplay'});
    
    visualGroup = matlab.system.display.SectionGroup(...
        'Title', 'Visualization', ...
        'Sections', visual);

    groups = [mainGroup realizationGroup visualGroup];
  end
end

end

% [EOF]