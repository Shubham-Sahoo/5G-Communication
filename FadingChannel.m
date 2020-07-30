classdef (Hidden) FadingChannel < matlab.System
%FadingChannel Filter input signal through a fading channel

% Copyright 2011-2019 The MathWorks, Inc.

%#codegen

% The following properties is defined in the child classes
%     FadingTechnique
%     NumSinusoids
%     InitialTimeSource
%     InitialTime
%     SampleRate;
%     PathDelays;
%     AveragePathGains;
%     MaximumDopplerShift;
%     DopplerSpectrum;
%     SpatialCorrelation;
%     NumTransmitAntennas;
%     NumReceiveAntennas;
%     TransmitCorrelationMatrix;
%     ReceiveCorrelationMatrix;
%     FadingDistribution;
%     KFactor;
%     mvalue;
%     DirectPathDopplerShift;
%     DirectPathInitialPhase;
%     Visualization;
%     AntennaPairsToDisplay;
%     PathsForDopplerDisplay;
%     SamplesToDisplay;

properties (Nontunable)
    %RandomStream Random number source
    %   Specify the source of random number stream as one of 'Global
    %   stream' | 'mt19937ar with seed'. If RandomStream is set to 'Global
    %   stream', the current global random number stream is used for
    %   normally distributed random number generation, in which case the
    %   reset method only resets the filters. If RandomStream is set to
    %   'mt19937ar with seed', the mt19937ar algorithm is used for normally
    %   distributed random number generation, in which case the reset
    %   method not only resets the filters but also re-initializes the
    %   random number stream to the value of the Seed property. The default
    %   value of this property is 'Global stream'.
    RandomStream = 'Global stream';  
    %Seed Initial seed
    %   Specify the initial seed of a mt19937ar random number generator
    %   algorithm as a double precision, real, nonnegative integer scalar.
    %   This property applies when you set the RandomStream property to
    %   'mt19937ar with seed'. The Seed is to re-initialize the mt19937ar
    %   random number stream in the reset method. The default value of this
    %   property is 73.
    Seed = 73;
end

properties (Nontunable, Logical)
    %NormalizePathGains Normalize average path gains to 0 dB
    %   Set this property to true to normalize the fading processes such
    %   that the total power of the path gains, averaged over time, is 0dB.
    %   The default value of this property is true.
    NormalizePathGains = true;
    %PathGainsOutputPort Output channel path gains
    %   Set this property to true to output the channel path gains of the
    %   underlying fading process. The default value of this property is
    %   false.
    PathGainsOutputPort = false;
end

properties (Access = protected, Nontunable)
    % Signal sample rate, only used in Simulink and derived from the
    % InheritSampleRate + SampleRate combination.    
    pSampleRate 
    % Cutoff frequency factor that is 1 by default for doppler.jakes
    pFcFactor = 1.0
end

properties (Access = protected)
    % Channel filter object
    cChannelFilter    
    % Channel filter info
    pChannelFilterInfo
    % Number of input samples that have been processed since the last reset
    pNumSamplesProcessed = 0
    % End time of last frame (for SOS burst mode only)
    pLastFrameEndTime = 0
end

properties (Access = private, Nontunable)
    % Number of transmit antennas
    pNt
    % Number of receive antennas
    pNr 
    % Number of paths, NP = length(PathDelays)
    pNumPaths      
    % Number of links, NL = pNt * pNr
    pNumLinks
    % Signal input data type
    pInputDataType = 'double'
    % Antenna selection flag
    pHasTxAntennaSelection
    % Antenna selection flag
    pHasRxAntennaSelection
    % Static channel flag
    pIsStaticChannel
    % Average path gain vector
    pAveragePathGainsLinear
    % Gaussian filter impulse response
    pGFilterImpulseResponse 
    % Polyphase filter flag
    pHasPolyphaseFilter
    % Polyphase filter bank
    pIFilterBank
    % Polyphase filter interpolation factor
    pIFilterInterpFactor
    % Linear filter flag
    pHasLinearFilter
    % Linear filter interpolation factor
    pLinearInterpFactor
    % FGN flag
    pIsFGN
    % Channel sampling time (for non-static channels only)
    pSampleTime
    % Discrete Doppler frequencies in GMEDS (for SOS non-static channels only)
    pDiscreteDopperReal
    pDiscreteDopperImag
    % SOS burst mode flag
    pIsSOSInputPortMode
    % Function handle for FGN or SOS generation
    pFadingTechniqueHandle
    % Column indices in 2-D path gain or Gaussian filter output for visualization
    pDisplayMIdx 
    % Impulse or frequency response visualization flag    
    pHasPathGainVis
    % Doppler spectrum visualization flag    
    pHasDopplerVis
    % Object for impulse and/or frequency response visualizations
    pPathGainVis
    % Object for Doppler spectrum visualization
    pDopplerVis
end

properties (Access = private)
    % Uncorrelated spatial correlation flag
    pIsRHIdentity
    % Spatial correlation matrices for all paths [NL, NL, NP] (for
    % spatially correlated channels only)
    pSQRTCorrelationMatrix
    % For Rician channel only 
    pLastThetaLOS
    % White Gaussian noise state
    pRNGStream
    % Gaussian filter state
    pGFilterState
    % Static channel path gains that stay constant for each input frame
    pStaticChannelPathGains
    % Polyphase filter bank phase
    pIFilterPhase
    % Polyphase filter state
    pIFilterState
    % Polyphase filter last outputs on new noise samples
    pIFilterNewSampLastOut
    % Polyphase filter last outputs
    pIFilterLastOutputs
    % Linear filter interpolation index
    pLinearInterpIndex
    % One-time sinusoid phases in GMEDS (for SOS only)
    pSinusoidsPhases
    % Time counter (for SOS non-static channels only)
    pTimeCounter
    % Fading process time offset (seconds) (for SOS non-static channels only)
    pInitialTime
end

properties(Constant, Access = private)
    % Polyphase filter length for path gain generation
    pPolyphaseFilterLength = 8
    % Fractional delay FIR filter length
    pChannelFilterLength = 16
    % Channel filter frac delay tolerance
    pFracDelayTolerance = 0.01
end

properties(Constant, Hidden)
    RandomStreamSet = matlab.system.StringSet({'Global stream', 'mt19937ar with seed'});
end

methods
  function obj = FadingChannel(varargin) % Constructor
    coder.allowpcode('plain');
    setProperties(obj, nargin, varargin{:});
  end  
    
  function set.Seed(obj, seed)
    propName = 'Seed';
    validateattributes(seed, {'double'}, ...
        {'real','scalar','integer','nonnegative','finite'}, ...
        [class(obj) '.' propName], propName);  %#ok<*EMCA>

    obj.Seed = seed;
  end
  
  function set.NormalizePathGains(obj, v)
    propName = 'NormalizePathGains';
    validateattributes(v, {'logical'}, {'scalar'}, ...
        [class(obj) '.' propName], propName); 

    obj.NormalizePathGains = v;
  end
  
  function set.PathGainsOutputPort(obj, v)
    propName = 'PathGainsOutputPort';
    validateattributes(v, {'logical'}, {'scalar'}, ...
        [class(obj) '.' propName], propName); 

    obj.PathGainsOutputPort = v;
  end    
end

methods(Access = protected)
  function num = getNumInputsImpl(obj)
    num = 1 + (strcmp(obj.AntennaSelection, 'Tx') || ...
               strcmp(obj.AntennaSelection, 'Tx and Rx')) ...
            + (strcmp(obj.AntennaSelection, 'Rx') || ...
               strcmp(obj.AntennaSelection, 'Tx and Rx')) ...
            + (strcmp(obj.FadingTechnique,  'Sum of sinusoids') && ...
              (strcmp(obj.InitialTimeSource,'Input port')));
  end
  
  function num = getNumOutputsImpl(obj)
    num = 1 + obj.PathGainsOutputPort;
  end  
  
  function validateInputsImpl(obj, x, varargin)
    % Validate data type and dimension in this method for all inputs,
    % except for the signal input dimension when Rx antenna selection is on
    validateattributes(x, {'double','single'}, {'2d','finite'}, ...
        class(obj), 'signal input'); 
                             
    [Nt, Nr] = getNumTxAndRx(obj);
    
    hasTxAntennaSelection = any(strcmp(obj.AntennaSelection, {'Tx', 'Tx and Rx'}));
    
    if hasTxAntennaSelection
        % Validate input for selected Tx
        validateattributes(varargin{1}, {'numeric'}, ...
            {'real','size',[1, Nt],'binary'}, class(obj), ...
            'input for selected transmit antennas');
    else
        % When the AntennaSelection property is set to 'Off' or 'Rx', the
        % signal input has a fixed number of columns. So we validate its
        % dimension here.     
        coder.internal.errorIf(size(x, 2) ~= getNumTxAndRx(obj), ...        
            'comm:FadingChannel:SignalInputNotMatchNumTx', 'transmit');    
    end
    
    if any(strcmp(obj.AntennaSelection, {'Rx', 'Tx and Rx'}))
        % Validate input for selected Rx
        validateattributes(varargin{hasTxAntennaSelection+1}, {'numeric'}, ...
            {'real','size',[1, Nr],'binary'}, class(obj), ...
            'input for selected receive antennas');
    end

    if strcmp(obj.FadingTechnique, 'Sum of sinusoids') && ...
        strcmp(obj.InitialTimeSource, 'Input port')
        % Validate init time input
        validateattributes(varargin{end}, {'double','single'}, ...
            {'real','scalar','finite','nonnegative'}, ...
            class(obj), 'initial time input');
    end
  end
  
  function setupImpl(obj, varargin)    
    % Set up system constants
    setupSystemParam(obj, varargin{1});

    % Set up RNG
    setupRNG(obj);
    
    % Set up spatial correlation matrix
    setupSQRTCorrelationMatrix(obj);
            
    if ~obj.pIsStaticChannel
        KI = getInterpFactors(obj);
    
        % Set up FGN or SOS source
        if obj.pIsFGN
            setupFGN(obj, KI);
        else
            setupSOS(obj, KI);
        end
        
        % Set up interpolation filters
        setupInterpFilters(obj, KI);
    end

    % Set up and initialize channel filter    
	setupChannelFilter(obj);
    
    % Set up visualization
    if isempty(coder.target) % No visualization in code generation
        setupVisualization(obj);
    end    
  end   
  
  function [Nt, Nr] = getNumTxAndRx(obj)
    switch obj.SpatialCorrelationSpecification
      case 'None'
        Nt = double(obj.NumTransmitAntennas);
        Nr = double(obj.NumReceiveAntennas);
      case 'Separate Tx Rx'
        Nt = size(obj.TransmitCorrelationMatrix, 1);
        Nr = size(obj.ReceiveCorrelationMatrix, 1);
      otherwise % 'Combined' 
        Nt = double(obj.NumTransmitAntennas);
        Nr = size(obj.SpatialCorrelationMatrix, 1)/Nt;
    end
  end
  
  function resetImpl(obj)
    inDT = obj.pInputDataType;
    
    % Reset object for visualization     
    if obj.pHasPathGainVis
        reset(obj.pPathGainVis);
    elseif obj.pHasDopplerVis && ~obj.pIsStaticChannel
        % No reset for static channels as we do not want to reset the scope
        % before each step call on it (no running mean calculation for the
        % empirical Doppler spectrum)
        reset(obj.pDopplerVis);
    end
    
    if strcmp(obj.FadingDistribution, 'Rician') && ~obj.pIsStaticChannel
        obj.pLastThetaLOS = ...
            cast(obj.DirectPathInitialPhase, inDT) / cast(2*pi, inDT) - ...
            cast(obj.DirectPathDopplerShift, inDT) / cast(obj.pSampleRate, inDT);
    end
    if strcmp(obj.FadingDistribution, 'nakagami') && ~obj.pIsStaticChannel
        obj.pLastThetaLOS = ...
            cast(obj.DirectPathInitialPhase, inDT) / cast(2*pi, inDT) - ...
            cast(obj.DirectPathDopplerShift, inDT) / cast(obj.pSampleRate, inDT);
        %disp("1st");
    end
        
    % Reset FGN or SOS source
    if obj.pIsFGN
        resetFGN(obj);
    else
        resetSOS(obj);
    end
         
    NL = obj.pNumLinks;
    NP = obj.pNumPaths;
    
    if obj.pIsStaticChannel   
        M  = NP * NL;
        % Get static paths gain for one sample
        if obj.pIsFGN
            w2 = generateRandn(obj, 1, 2*M);        
            wgnoise = 1/sqrt(2) * ...
                (w2(1, 1:M) + 1i*w2(1, M+1:end)); % [1, M]
        else
            N1 = obj.NumSinusoids;
            temp = cos(obj.pSinusoidsPhases); % [N1, 1, NP, Nt, Nr]
            wgnoise = sqrt(1/N1) * reshape(sum( ...
                complex(temp(1:N1,:,:,:,:), temp(N1+1:end,:,:,:,:)), 1), ...
                1, []); % [1, M]            
        end

        if obj.pHasTxAntennaSelection || obj.pHasRxAntennaSelection
            obj.pStaticChannelPathGains = wgnoise;
        else % Scale static path gains
            obj.pStaticChannelPathGains = scalePathGains(obj, wgnoise); % [1, M]
        end
        
        if obj.pHasDopplerVis && ~isScopeClosed(obj.pDopplerVis)
            % Update Doppler spectrum visualization. 
            step(obj.pDopplerVis, ...
                double(abs(wgnoise(:, obj.pDisplayMIdx(obj.PathsForDopplerDisplay))).^2));
        end
    end

    % Reset channel filter
    reset(obj.cChannelFilter);
    
    obj.pNumSamplesProcessed = 0;
    
  end
  
  function varargout = stepImpl(obj, x, varargin)
    Ns = size(x, 1);
    Nt = obj.pNt;
    Nr = obj.pNr;
    NP = obj.pNumPaths;    
        
    % Validate init time input value: Must be positive integer multiple of
    % input sample time and larger than where the last frame stops
    if obj.pIsSOSInputPortMode
        Rs = obj.pSampleRate;
        initTime = double(varargin{end});
        sampPos = initTime*Rs;
        lastSampPos = round(obj.pLastFrameEndTime*Rs);
        coder.internal.errorIf( ...
            any(isinf(initTime)) || any(isnan(initTime)) || ...
        	(round(sampPos) < lastSampPos), ...
            'comm:FadingChannel:InvalidInitTimeInput', ...
            feval( 'sprintf' , '%0.8g', lastSampPos/Rs));
         
        if Ns > 0 % Not log anything for empty frames - just ignore them
            roundedInitTime = round(initTime*Rs)/Rs;
            updateFiltersBurst(obj, roundedInitTime, Ns);            
        end
    end
    
    % Validate the SELTX input and use it to check against the number
    % of columns of the signal input.
    if obj.pHasTxAntennaSelection
        txSelIn = varargin{1};
        coder.internal.errorIf(any((txSelIn ~= 0) & ...
            (txSelIn ~= 1)) || (sum(txSelIn) == 0), ...
            'comm:FadingChannel:InvalidSelectedTxInput');
        coder.internal.errorIf(size(x, 2) ~= sum(txSelIn), ...
            'comm:FadingChannel:SignalInputNotMatchNumTx', ...
            'selected transmit');        
        activeTx = logical(txSelIn);
    else
        activeTx = true(1, Nt);
    end
        
    % Validate the SELRX input
    if obj.pHasRxAntennaSelection
        rxSelIn = varargin{1 + obj.pHasTxAntennaSelection}; 
        coder.internal.errorIf(any((rxSelIn ~= 0) & ...
            (rxSelIn ~= 1)) || (sum(rxSelIn) == 0), ...
            'comm:FadingChannel:InvalidSelectedRxInput');
        activeRx = logical(rxSelIn);        
    else
        activeRx = true(1, Nr);
    end
    numActiveRx = sum(activeRx);
        
    % Output empty signals and path gains for an empty input
    if Ns == 0
        varargout{1} = zeros(Ns, numActiveRx, 'like', x);
        if obj.PathGainsOutputPort
            varargout{2} = NaN(Ns, NP, Nt, Nr, 'like', x);
        end
        return;
    end
    
    % Generate path gains for all links, NOT only those active links.
    if ~obj.pIsStaticChannel
        z = generatePathGains(obj, Ns);             % [Ns, M]
        if obj.pHasTxAntennaSelection || obj.pHasRxAntennaSelection
            % Spatial correlation only applied to the active links
            activeLIdx = find(kron(activeTx, activeRx) == 1);
            z = scalePathGains(obj, z, activeLIdx);     % [Ns, M]
        else
            z = scalePathGains(obj, z);     % [Ns, M]
        end
    elseif obj.pHasTxAntennaSelection || obj.pHasRxAntennaSelection
        % Spatial correlation only applied to the active links
        activeLIdx = find(kron(activeTx, activeRx) == 1);
        % Scale static path gains. 
        z = scalePathGains(obj, obj.pStaticChannelPathGains, ...
            activeLIdx);                            % [1, M]
    else % Use previously saved scaled path gains for static channels
        z = obj.pStaticChannelPathGains;            % [1, M]
    end
    
    if obj.NormalizeChannelOutputs
        % Normalize by the number of selected receive antennas so that the
        % total output power is equal to the total input power
        z = z/sqrt(numActiveRx);
    end
    
    % Reshape z ready for channel filtering
    g = reshape(z, [], Nr, Nt, NP);

    % Filter signal input
    if obj.pHasTxAntennaSelection 
        % Fill with zero signals for Tx not selected
        packedX = zeros(Ns, Nt, 'like', x);
        % Write the RHS of the following line this way to work around
        % codegen limitation
        packedX(:, activeTx) = x(:,1:sum(activeTx)); 
        varargout{1} = step(obj.cChannelFilter, packedX, g(:,activeRx,:,:)); 
    elseif obj.pHasRxAntennaSelection
        varargout{1} = step(obj.cChannelFilter, x, g(:,activeRx,:,:)); 
    else 
        varargout{1} = step(obj.cChannelFilter, x, g);  
    end
    
    % Postprocessing: Modify path gains ready for output and/or visual
    if (obj.PathGainsOutputPort || obj.pHasPathGainVis || obj.pHasDopplerVis) && ...
        (obj.pHasTxAntennaSelection || obj.pHasRxAntennaSelection)
        g(:, ~activeRx, :, :) = NaN('like', x);
        g(:, :, ~activeTx, :) = NaN('like', x);
    end

    % Reshape channel path gains to the 4-D format of Ns x NP x Nt x Nr
    if obj.PathGainsOutputPort
        % We have to conditionally determine the output dimension according
        % to the value of Nt and Nr because of a limitation of System block
        % in code generation for variable-size n-D signals. For
        % example, when Nt == 1 and/or Nr == 1, the object propagator
        % generates a 2-D/3-D output, but coder constantly infers a 4-D
        % output if we do not manually do the following (the last 'else'
        % branch is actually sufficient to handle all the cases in MATLAB
        % and normal mode simulation for System block).

        if (Nr == 1) && (Nt == 1)
            pathGains = reshape(g, [], NP);
        elseif Nr == 1 % Can be simplified.
            pathGains = permute(reshape(g, [], Nt, NP), [1 3 2]);
        else
            pathGains = permute(g, [1, 4, 3, 2]);
        end
        
        if obj.pIsStaticChannel
            varargout{2} = repmat(pathGains, [Ns 1]);
        else
            varargout{2} = pathGains;
        end
    end

    % Update samples that have been processed
    obj.pNumSamplesProcessed = obj.pNumSamplesProcessed + Ns;
    
    % Visualize current frame
    if obj.pHasPathGainVis || obj.pHasDopplerVis        
        if obj.pIsStaticChannel            
            % g may not exist for doppler
            channelDisplay(obj, repmat(g(:)', [Ns 1])); 
        else
            channelDisplay(obj, reshape(g, Ns, []));
        end        
    end
  end
  
  function releaseImpl(obj)
    release(obj.cChannelFilter);

    if obj.pHasPathGainVis
        release(obj.pPathGainVis);
    elseif obj.pHasDopplerVis
        release(obj.pDopplerVis);
    end
  end
    
  function flag = isInactivePropertyImpl(obj, prop)
    % Use the if-else format for codegen
    if strcmp(prop, 'Seed')
        flag = strcmp(obj.RandomStream, 'Global stream');
    elseif strcmp(prop, 'PathsForDopplerDisplay')
        flag = ~strcmp(obj.Visualization, 'Doppler spectrum');
    elseif strcmp(prop, 'SamplesToDisplay')
        flag = strcmp(obj.Visualization, 'Off') || ...
               strcmp(obj.Visualization, 'Doppler spectrum'); 
    elseif strcmp(prop, 'InitialTime') 
         flag = (strcmp(obj.InitialTimeSource, 'Input port') || strcmp(obj.FadingTechnique, 'Filtered Gaussian noise'));
    elseif strcmp(prop, 'InitialTime') || strcmp(prop, 'InitialTimeSource') ...
            || strcmp(prop, 'NumSinusoids')
        flag = strcmp(obj.FadingTechnique, 'Filtered Gaussian noise');
    else 
        flag = false;
    end
  end
  
  function s = saveObjectImpl(obj)
    s = saveObjectImpl@matlab.System(obj);
    if isLocked(obj)
        s.pIsRHIdentity = obj.pIsRHIdentity;
        s.pFcFactor = obj.pFcFactor;
        s.pSQRTCorrelationMatrix = obj.pSQRTCorrelationMatrix;         
        s.pNumSamplesProcessed = obj.pNumSamplesProcessed;        
        s.pSampleRate = obj.pSampleRate;
        s.pNt = obj.pNt;
        s.pNr = obj.pNr;
        s.pNumPaths = obj.pNumPaths;      
        s.pNumLinks = obj.pNumLinks;
        s.pInputDataType = obj.pInputDataType;
        s.pHasTxAntennaSelection = obj.pHasTxAntennaSelection;
        s.pHasRxAntennaSelection = obj.pHasRxAntennaSelection;
        s.pIsStaticChannel = obj.pIsStaticChannel;
        s.pAveragePathGainsLinear = obj.pAveragePathGainsLinear;
        s.pGFilterImpulseResponse = obj.pGFilterImpulseResponse; 
        s.pHasPolyphaseFilter = obj.pHasPolyphaseFilter;
        s.pIFilterBank = obj.pIFilterBank; 
        s.pIFilterInterpFactor = obj.pIFilterInterpFactor;
        s.pHasLinearFilter = obj.pHasLinearFilter;
        s.pLinearInterpFactor = obj.pLinearInterpFactor;
        s.pLastThetaLOS = obj.pLastThetaLOS;
        s.pRNGStream = obj.pRNGStream;
        s.pGFilterState = obj.pGFilterState;
        s.pStaticChannelPathGains = obj.pStaticChannelPathGains;
        s.pIFilterPhase = obj.pIFilterPhase;
        s.pIFilterState = obj.pIFilterState;
        s.pIFilterNewSampLastOut = obj.pIFilterNewSampLastOut;
        s.pIFilterLastOutputs = obj.pIFilterLastOutputs;
        s.pLinearInterpIndex = obj.pLinearInterpIndex;
        s.pHasPathGainVis = obj.pHasPathGainVis;
        s.pHasDopplerVis = obj.pHasDopplerVis;
        s.pDisplayMIdx = obj.pDisplayMIdx;
        s.pIsFGN = obj.pIsFGN;
        s.pIsSOSInputPortMode = obj.pIsSOSInputPortMode;
        s.pTimeCounter = obj.pTimeCounter;
        s.pSinusoidsPhases = obj.pSinusoidsPhases;
        s.pFadingTechniqueHandle = obj.pFadingTechniqueHandle;
        s.pSampleTime = obj.pSampleTime;
        s.pDiscreteDopperReal = obj.pDiscreteDopperReal;
        s.pDiscreteDopperImag = obj.pDiscreteDopperImag;
        s.pLastFrameEndTime = obj.pLastFrameEndTime;
        s.pInitialTime = obj.pInitialTime;
        s.pChannelFilterInfo = obj.pChannelFilterInfo;        
        s.cChannelFilter = matlab.System.saveObject(obj.cChannelFilter);
        s.pPathGainVis = matlab.System.saveObject(obj.pPathGainVis);
        s.pDopplerVis = matlab.System.saveObject(obj.pDopplerVis);
    end
  end
  
  function loadObjectImpl(obj, s, wasLocked)
    if wasLocked
        obj.pIsRHIdentity = s.pIsRHIdentity;
        obj.pFcFactor = s.pFcFactor;
        obj.pSQRTCorrelationMatrix = s.pSQRTCorrelationMatrix; 
        obj.pNt = s.pNt;
        obj.pNr = s.pNr;
        obj.pNumPaths = s.pNumPaths;      
        obj.pNumLinks = s.pNumLinks;
        obj.pHasTxAntennaSelection = s.pHasTxAntennaSelection;
        obj.pHasRxAntennaSelection = s.pHasRxAntennaSelection;
        obj.pIsStaticChannel = s.pIsStaticChannel;
        obj.pAveragePathGainsLinear = s.pAveragePathGainsLinear;
        obj.pGFilterImpulseResponse = s.pGFilterImpulseResponse; 
        obj.pHasPolyphaseFilter = s.pHasPolyphaseFilter;
        obj.pIFilterBank = s.pIFilterBank; 
        obj.pIFilterInterpFactor = s.pIFilterInterpFactor;
        obj.pLinearInterpFactor = s.pLinearInterpFactor;
        obj.pLastThetaLOS = s.pLastThetaLOS;
        obj.pGFilterState = s.pGFilterState;
        obj.pStaticChannelPathGains = s.pStaticChannelPathGains;
        obj.pIFilterPhase = s.pIFilterPhase;
        obj.pIFilterState = s.pIFilterState;
        obj.pIFilterNewSampLastOut = s.pIFilterNewSampLastOut;
        obj.pIFilterLastOutputs = s.pIFilterLastOutputs;
        obj.pLinearInterpIndex = s.pLinearInterpIndex;

        % New properties from R2014b
        if isfield(s, 'pHasPathGainVis') 
            obj.pNumSamplesProcessed = s.pNumSamplesProcessed;
            obj.pIsFGN = s.pIsFGN;
            obj.pIsSOSInputPortMode = s.pIsSOSInputPortMode;
            obj.pHasPathGainVis = s.pHasPathGainVis;
            obj.pHasDopplerVis = s.pHasDopplerVis;
            obj.pDisplayMIdx = s.pDisplayMIdx;
            obj.pPathGainVis = matlab.System.loadObject(s.pPathGainVis);
            obj.pDopplerVis = matlab.System.loadObject(s.pDopplerVis);
        end
        
        if isfield(s, 'pTimeCounter')  
            obj.pTimeCounter = s.pTimeCounter;
            obj.pSinusoidsPhases = s.pSinusoidsPhases;
            obj.pFadingTechniqueHandle = s.pFadingTechniqueHandle;
            obj.pDiscreteDopperReal = s.pDiscreteDopperReal;
            obj.pDiscreteDopperImag = s.pDiscreteDopperImag;
            obj.pInitialTime = s.pInitialTime;
        end
        
        % New properties from R2017b
        if isfield(s, 'pWGNState') && ~isempty(s.pWGNState)
            obj.pRNGStream = RandStream('mt19937ar');
            obj.pRNGStream.State = s.pWGNState;
        elseif isfield(s, 'pRNGStream') && ~isempty(s.pRNGStream)
            obj.pRNGStream = RandStream(obj.RandomNumGenerator, 'Seed', obj.Seed);
            obj.pRNGStream.State = s.pRNGStream.State;
        end        
                
        % Property rename in R2017b
        % pLinearFilter --> pHasLinearFilter
        if isfield(s, 'pLinearFilter') 
            obj.pHasLinearFilter = s.pLinearFilter;
        elseif isfield(s, 'pHasLinearFilter')
            obj.pHasLinearFilter = s.pHasLinearFilter;            
        end
        
        % pTs --> pSampleTime
        if isfield(s, 'pTs') 
            obj.pSampleTime = s.pTs;
        elseif isfield(s, 'pSampleTime')
            obj.pSampleTime = s.pSampleTime;
        end
        
        % pNumLastBurstSamples & pLastFrameTime --> pLastFrameEndTime
        if isfield(s, 'pLastFrameEndTime')             
            obj.pLastFrameEndTime = s.pLastFrameEndTime;
        elseif isfield(s, 'pNumLastBurstSamples')
            obj.pLastFrameEndTime = ...
                s.pNumLastBurstSamples/s.SampleRate + s.pLastFrameTime;
        end
        
        % New properties in R2018a
        if isfield(s, 'pSampleRate')
            obj.pSampleRate = s.pSampleRate;
            obj.pInputDataType = s.pInputDataType;
        end
        
        if isfield(s, 'cChannelFilter')
            obj.cChannelFilter = matlab.System.loadObject(s.cChannelFilter);            
            obj.pChannelFilterInfo = s.pChannelFilterInfo;
        else % Object saved prior to 19b. Instantiate a channel filter
            if isfield(s, 'PathDelays')
                pd = s.PathDelays;
            elseif isfield(s, 'Profile') % comm.LTEMIMOChannel
                switch s.Profile(1:3)
                  case 'EPA'
                    pd = [0 30 70 90 110 190 410]*1e-9;
                  case 'EVA'
                    pd = [0 30 150 310 370 710 1090 1730 2510]*1e-9;
                  otherwise % 'ETU'
                    pd = [0 50 120 200 230 500 1600 2300 5000]*1e-9;
                end
            end
            obj.cChannelFilter = comm.internal.channel.ChannelFilter( ...
                'SampleRate', s.SampleRate, ...                
                'PathDelays', pd);
        end
    end
    loadObjectImpl@matlab.System(obj, s);
  end
  
  function flag = isInputSizeLockedImpl(~,~)
     flag = false;
  end  
end

methods(Access = private) % General methods
  function setupSystemParam(obj, x)
    % Set sample rate. In Simulink, the pSampleRate has bee set in the
    % child object's setupImpl, dependent on the 'InheritSampleRate' and
    % 'SampleRate' property combination. So here we only need to set
    % pSampleRate if in MATLAB.
    if (getExecPlatformIndex(obj) ~= 1)    
        obj.pSampleRate = obj.SampleRate;
    end
    
    % Set Nt, Nr, NP, NL  
    [obj.pNt, obj.pNr] = getNumTxAndRx(obj);    % Nt & Nr
    obj.pNumPaths = length(obj.PathDelays);     % NP
    obj.pNumLinks = obj.pNt * obj.pNr;          % NL

    % Signal input data type: double or single
    obj.pInputDataType = class(x);
    
    % Set static channel flag
    obj.pIsStaticChannel = (obj.MaximumDopplerShift == 0);
    
    % Set FGN/SOS flags    
    obj.pIsFGN = strcmp(obj.FadingTechnique, 'Filtered Gaussian noise');
    
    % Set initial time mode flag for SOS
    obj.pIsSOSInputPortMode = ~obj.pIsFGN && ...
        strcmp(obj.InitialTimeSource, 'Input port');
    
    % Set antenna selection flags
    obj.pHasTxAntennaSelection = strcmp(obj.AntennaSelection, 'Tx') || ...
                                 strcmp(obj.AntennaSelection, 'Tx and Rx');

    obj.pHasRxAntennaSelection = strcmp(obj.AntennaSelection, 'Rx') || ...
                                 strcmp(obj.AntennaSelection, 'Tx and Rx');      

    % Set visualization flags
    obj.pHasPathGainVis = ~strcmp(obj.Visualization, 'Off') && ...
                          ~strcmp(obj.Visualization, 'Doppler spectrum');
    obj.pHasDopplerVis  =  strcmp(obj.Visualization, 'Doppler spectrum');    
    
    % Calculate pAveragePathGainsLinear of size [1, M]
    PP = (10.^(obj.AveragePathGains/10));
    if obj.NormalizePathGains
        obj.pAveragePathGainsLinear = kron(sqrt((PP)/sum(PP)), ...
            ones(1, obj.pNumLinks, 'like', x));  
    else 
        obj.pAveragePathGainsLinear = kron(sqrt(PP), ...
            ones(1, obj.pNumLinks, 'like', x)); 
    end
  end
        
  function setupSQRTCorrelationMatrix(obj)
    NL   = obj.pNumLinks;
    inDT = obj.pInputDataType;
    
    if strcmp(obj.SpatialCorrelationSpecification, 'None')
        obj.pIsRHIdentity = true;        
        if ~isempty(coder.target) % Not used, only for codegen
            obj.pSQRTCorrelationMatrix = coder.nullcopy( ...
                complex(eye(NL, inDT))); 
        end
    else
        NP = obj.pNumPaths;
        NL = obj.pNumLinks;
                
        if strcmp(obj.SpatialCorrelationSpecification, 'Separate Tx Rx')    
            Rt = cast(obj.TransmitCorrelationMatrix, inDT);
            Rr = cast(obj.ReceiveCorrelationMatrix, inDT);
            if ismatrix(Rt) && ismatrix(Rr)
                RH = kron(Rt, Rr);
                if obj.NormalizePathGains
                    RH = RH*NL/trace(RH);
                end
            else
                RH = complex(zeros(NL, NL, NP, inDT));
                for n = coder.unroll(1:NP)
                    if ismatrix(Rt)
                        thisRt = Rt; 
                    else
                        thisRt = Rt(:,:,n); 
                    end

                    if ismatrix(Rr)
                        thisRr = Rr; 
                    else
                        thisRr = Rr(:,:,n);
                    end
                    
                    RH(:,:,n) = kron(thisRt, thisRr);
                    if obj.NormalizePathGains                       
                        RH(:,:,n) = RH(:,:,n) * (NL/trace(RH(:,:,n)));
                    end
                end
            end
        else % Combined
            RH = cast(obj.SpatialCorrelationMatrix, inDT);
            if obj.NormalizePathGains
                if ismatrix(RH)
                    RH = RH*NL/trace(RH);
                else
                    for n = coder.unroll(1:NP)
                        RH(:,:,n) = RH(:,:,n) * (NL/trace(RH(:,:,n)));
                    end
                end
            end
        end
        
        if ismatrix(RH)
            obj.pIsRHIdentity = isequal(RH, eye(NL, inDT));     
            if ~obj.pIsRHIdentity
                if obj.pHasTxAntennaSelection || obj.pHasRxAntennaSelection
                    obj.pSQRTCorrelationMatrix = RH.';
                else
                    obj.pSQRTCorrelationMatrix = sqrtm(RH.');
                end
            elseif ~isempty(coder.target) % Not used, only for codegen
                obj.pSQRTCorrelationMatrix = coder.nullcopy( ...
                    complex(eye(NL, inDT)));
            end
        else % 3D RH    
            obj.pIsRHIdentity = false(1, NP);
            obj.pSQRTCorrelationMatrix = complex(zeros(NL, NL, NP, inDT));
            
            for n = coder.unroll(1:NP)
                thisRH = RH(:,:,n);
                obj.pIsRHIdentity(1, n) = isequal(thisRH, eye(NL, inDT));
            
                if ~obj.pIsRHIdentity(1, n)
                    if obj.pHasTxAntennaSelection || obj.pHasRxAntennaSelection
                        obj.pSQRTCorrelationMatrix(:,:,n) = thisRH.'; 
                    else
                        obj.pSQRTCorrelationMatrix(:,:,n) = sqrtm(thisRH.'); 
                    end
                end
            end
        end
    end
  end

  function KI = getInterpFactors(obj)
    Rs = obj.pSampleRate;
    maxDopplerShift = obj.MaximumDopplerShift;
    fcFactor = obj.pFcFactor;
    
    % Check the restriction between MaximumDopplerShift and SampleRate
    targetOversampleFactor = 10 * fcFactor; 
    targetOversampledDoppler = maxDopplerShift * targetOversampleFactor; 
    coder.internal.errorIf(any(targetOversampledDoppler > Rs), ...
        'comm:FadingChannel:MaxDopplerShiftTooLarge', ...
        coder.const(sprintf('%1.5g', max(targetOversampleFactor))));
        
    % Polyphase and linear interpolation factors correspond to the fading
    % process with the highest bandwidth.
    Ks1min = 10;
    Ks1max = 20;
    [~, ifcmax] = max(targetOversampledDoppler);
    for i = coder.unroll(1:length(obj.pFcFactor))
        Ks = floor(Rs/targetOversampledDoppler(i));
        if Ks <= Ks1max
            Ks0 = Ks;
            Ks1 = Ks;
            Ks2 = 1;
        else
            Ks1 = Ks1min;
            Ks2 = round(Ks/Ks1);
            Ks0 = Ks1 * Ks2;
        end

        coder.internal.errorIf((10*Ks0*targetOversampledDoppler(i)) < Rs, ...
            'comm:FadingChannel:OverSampFactorTooLarge');

        if i == ifcmax
            KI = [Ks0, Ks1, Ks2];
        end
    end
  end
  
  function g = scalePathGains(obj, z, varargin)
    NP = obj.pNumPaths;
    NL = obj.pNumLinks;
    Ns = size(z, 1); % z is of size [Ns, M]
    inDT = obj.pInputDataType;
    
    % Apply spatial correlation to the active links ONLY. Considering
    % spatial correlation is an attribute of antennas and an inactive
    % antenna cannot contribute to the active ones, we cannot apply the
    % whole correlation matrix to all the links by doing z = z *
    % obj.pSQRTCorrelationMatrix.    
    if ~all(obj.pIsRHIdentity) && ((nargin == 2) || ...
            ((nargin == 3) && length(varargin{1}) > 1))
        if nargin == 2
            corrMtx = obj.pSQRTCorrelationMatrix;
        else
            activeLIdx = varargin{1};
            corrMtx = obj.pSQRTCorrelationMatrix(activeLIdx, activeLIdx, :);
        end
        
        for n = 1:NP
            if isscalar(obj.pIsRHIdentity)
                thisCorrMtx = corrMtx;
            elseif ~obj.pIsRHIdentity(n)
                thisCorrMtx = corrMtx(:, :, n);
            else % No correlation for this path
                continue;
            end
            
            if obj.pHasTxAntennaSelection || obj.pHasRxAntennaSelection
                thisCorrMtx = sqrtm(thisCorrMtx);
            end
            
            if nargin == 2
                z(:, (n-1)*NL+1:n*NL) = z(:, (n-1)*NL+1:n*NL) * thisCorrMtx;
            else
                z(:, (n-1)*NL+activeLIdx) = z(:, (n-1)*NL+activeLIdx) * thisCorrMtx;
            end
        end
    end
    
    if strcmp(obj.FadingDistribution, 'Rician')
        % Number of Rician fading links, M or NL
        numRLinks = length(obj.KFactor) * NL; 
        
        if ~obj.pIsStaticChannel
            theta = cast(obj.DirectPathDopplerShift/obj.pSampleRate, inDT);

            % The starting phase is equal to the phase at the end
            % of the last frame plus a phase increment of theta.
            thetaInitLOS = obj.pLastThetaLOS + theta;

            % Phase offsets for the LOS path(s). No offset for first sample:
            % start at time t = 0, i.e., exp(jw0) = 1
            p = cumsum([thetaInitLOS; repmat(theta, Ns-1, 1)], 1);

            % Store last values of phase offsets for next iteration
            obj.pLastThetaLOS = p(end, :);

            t = reshape(repmat(exp(1i*2*pi*p), [NL, 1]), [Ns, numRLinks]); % [Ns, numRLinks]
        else
            t = ones(inDT);
        end
        
        K = reshape(repmat(cast(obj.KFactor, inDT), [Ns*NL, 1]), ...
            [Ns, numRLinks]); % [Ns, numRLinks]

        z(:,1:numRLinks) = (z(:,1:numRLinks) + t.*sqrt(K)) ./ sqrt(K+1);
    
    end
    if strcmp(obj.FadingDistribution, 'nakagami')
        % Number of Rician fading links, M or NL
        numRLinks = length(obj.KFactor) * NL; 
        disp("2nd here");
        if ~obj.pIsStaticChannel
            theta = cast(obj.DirectPathDopplerShift/obj.pSampleRate, inDT);

            % The starting phase is equal to the phase at the end
            % of the last frame plus a phase increment of theta.
            thetaInitLOS = obj.pLastThetaLOS + theta;

            % Phase offsets for the LOS path(s). No offset for first sample:
            % start at time t = 0, i.e., exp(jw0) = 1
            p = cumsum([thetaInitLOS; repmat(theta, Ns-1, 1)], 1);

            % Store last values of phase offsets for next iteration
            obj.pLastThetaLOS = p(end, :);
            
            t = reshape(repmat(exp(1i*2*pi*p), [NL, 1]), [Ns, numRLinks]); % [Ns, numRLinks]
        else
            t = ones(inDT);
        end
        
        K = reshape(repmat(cast(obj.KFactor, inDT), [Ns*NL, 1]), ...
            [Ns, numRLinks]); % [Ns, numRLinks]
        m = obj.mvalue;
        z(:,1:numRLinks) = (z(:,1:numRLinks)).*exp(1-m) + ((z(:,1:numRLinks) + t.*sqrt(K)) ./ sqrt(K+1)).*(1-exp(1-m));
        %z(:,1:numRLinks) = (z(:,1:numRLinks) + t.*sqrt(K)) ./ sqrt(K+1);
    end
    
    % Apply path gains
    g = bsxfun(@times, obj.pAveragePathGainsLinear, z);
  end   
end

methods(Access = private) % FNG related methods
  function setupFGN(obj, KI)
    % This method is called only for non-static channels
    coder.extrinsic('FadingChannel.getGaussianFilterIR');  
    
    % Use minimum sampling period across all Doppler spectra, corresponding
    % to the largest cutoff frequency. Force all impulse responses to be of
    % equal length.
    fc = obj.MaximumDopplerShift .* obj.pFcFactor;
    fcmin = min(fc);
    fgTs  = KI(1)/obj.pSampleRate;
    t     = -50/(2*pi*fcmin):fgTs:50/(2*pi*fcmin);

    obj.pGFilterImpulseResponse = coder.const(...
        cast(FadingChannel.getGaussianFilterIR( ...
        fgTs, fc, t, obj.DopplerSpectrum), obj.pInputDataType));
  end
  
  function resetFGN(obj)
    NP = obj.pNumPaths;
    NL = obj.pNumLinks;
    M  = NP * NL;
    
    % Reset random number generator
    resetRNG(obj);

    if ~obj.pIsStaticChannel
        % Flush Gaussian filter with initial hLen noise samples
        hLen = size(obj.pGFilterImpulseResponse, 2);
        obj.pGFilterState = complex(zeros(hLen-1, M, obj.pInputDataType));
        FGNGenerateOutput(obj, hLen);
        
        if obj.pHasPolyphaseFilter 
            % Initialize interpolation filter phase and indices
            initInterpFilters(obj);
            % Reset polyphase filters
            resetPolyphaseFilters(obj);
        else
            % Flush filters with one sample
            FGNGenerateOutput(obj, 1);
        end
       
        generatePathGains(obj, 1);
    end
  end  
  
  function y = FGNGenerateOutput(obj, N)
    NP = obj.pNumPaths;
    NL = obj.pNumLinks;
    M  = NP * NL;

    % Gaussian noise generation
    w2 = generateRandn(obj, N, 2*M); % [N, 2M]

    % Complex noise of size [N, M]
    wgnoise = 1/sqrt(2)*(w2(:,1:M) + 1i*w2(:,M+1:end)); 

    % Impulse response of filter.    
    IR = obj.pGFilterImpulseResponse;
    fcLen = size(IR, 1); % fcLen = 1 or NP
    
    if fcLen == 1
        [y, obj.pGFilterState] = filter(IR, 1, wgnoise, obj.pGFilterState, 1);
    else % fcLen == NP
        y = coder.nullcopy(complex(zeros(N, M, obj.pInputDataType)));
        filterState = obj.pGFilterState;
        for i = coder.unroll(1:fcLen)
            colIdx = (i-1)*NL+(1:NL); % Indices of links for path i
            [y(:,colIdx), filterState(:,colIdx)] = ...
                filter(IR(i,:), 1, wgnoise(:,colIdx), filterState(:,colIdx), 1);
        end        
        obj.pGFilterState = filterState;
    end
    
    if obj.pHasDopplerVis && ~isScopeClosed(obj.pDopplerVis) 
        % Update Doppler spectrum visualization. Note that this function is
        % not hit in a static channel configuration.
        step(obj.pDopplerVis, ...
            double(y(:, obj.pDisplayMIdx(obj.PathsForDopplerDisplay))));
    end
  end
end

methods(Access = private) % SOS related methods 
  function setupSOS(obj, KI)
    % This method is called only for non-static channels
    NP = obj.pNumPaths;
    N1 = obj.NumSinusoids;         

    obj.pInitialTime = 0;
    
    % Set SOS sampling time
    obj.pSampleTime = KI(1)/obj.pSampleRate;

    % Initialize GMEDS discrete Doppler frequencies 
    temp1 = (pi/(2*N1))*((1:N1)'-0.5);
    temp2 = (1:NP)*pi/(4*N1*(NP+2));

    maxDopplerShift = cast(obj.MaximumDopplerShift, obj.pInputDataType);
    obj.pDiscreteDopperReal = reshape(2 * pi * maxDopplerShift * ... 
        cos(bsxfun(@plus,  temp1, temp2)), [N1, 1, NP]); % [N1 1 NP]
    obj.pDiscreteDopperImag = reshape(2 * pi * maxDopplerShift * ... 
        cos(bsxfun(@minus, temp1, temp2)), [N1, 1, NP]); % [N1 1 NP]
  end
  
  function resetSOS(obj)
    % Reset random number generator
    resetRNG(obj);
    
    % Generate starting phases of the sinusoids in GMEDS
    generateSinusoidsPhases(obj);
    
    if ~obj.pIsStaticChannel 
        % The pTimeCounter is used to track the number of SOS samples from
        % the start of the simulation.
        obj.pTimeCounter = 0;      
        
        if ~obj.pIsSOSInputPortMode
            Rs = obj.pSampleRate;
            initTime = round(obj.InitialTime*Rs)/Rs;
            obj.pInitialTime = initTime;

            % Use InitialTime property to bring interpolation filter states
            % and indices up to the initial time
            updateInterpFilters(obj);

            % Estimate the Rician phases based on the InitialTime value
            if strcmp(obj.FadingDistribution, 'Rician')
                updateRicianLOSFhase(obj, initTime);
            end
            if strcmp(obj.FadingDistribution, 'nakagami')
                updateRicianLOSFhase(obj, initTime);
                disp("3rd here");
            end
        end
    end
    
    if obj.pIsSOSInputPortMode   
        obj.pLastFrameEndTime = 0;
    end
  end
  
  function updateFiltersBurst(obj, initTime, Ns)
    % Log initial time input
    obj.pInitialTime = initTime; 

    if ~obj.pIsStaticChannel
        % Bring interpolation filter states and indices up to the initial time
        updateInterpFilters(obj)

        % Update Rician phase for a non-continuous frame
        if (initTime > obj.pLastFrameEndTime) && ...
            strcmp(obj.FadingDistribution, 'Rician')            
            updateRicianLOSFhase(obj, initTime - obj.pLastFrameEndTime);
        end
        if (initTime > obj.pLastFrameEndTime) && ...
            strcmp(obj.FadingDistribution, 'nakagami')            
            updateRicianLOSFhase(obj, initTime - obj.pLastFrameEndTime);
            disp("4th here");
        end
    end
    
    % Update channel reset flags for a non-continuous frame
    if (initTime > obj.pLastFrameEndTime)      
        reset(obj.cChannelFilter);
    end
    
    obj.pLastFrameEndTime = initTime + Ns/obj.pSampleRate;
  end  
  
  function y = SOSGenerateOutput(obj, N) 
    N1 = obj.NumSinusoids;
    P  = obj.pSinusoidsPhases;    
    C  = obj.pTimeCounter;

    % The assertion is needed in order to put a bound on N before this
    % assignment to t. The assertion is needed for codegen. 
    assert(N < N+1);
    
    % Get time instances 
    t = obj.pInitialTime + ...
        ((C-obj.pPolyphaseFilterLength) + (0:(N-1)))*obj.pSampleTime; % [1, N]

    % Discrete Doppler phases at the time instances
    D1 = bsxfun(@times, obj.pDiscreteDopperReal, t); % [N1 N NP]
    D2 = bsxfun(@times, obj.pDiscreteDopperImag, t); % [N1 N NP]
    
    % Add to the initial phases and sum all sinusoids 
    y = sqrt(1/N1) * reshape(sum(complex( ...
        cos(bsxfun(@plus, P(1:N1,:,:,:,:),     D1)), ...  % [N1 N NP Nt Nr]
        cos(bsxfun(@plus, P(N1+1:end,:,:,:,:), D2))), ... % [N1 N NP Nt Nr]
        1), N, []); % [N M]

    % Update time counter
    obj.pTimeCounter = C + N;
  end   
  
  function generateSinusoidsPhases(obj)
    % Generate the initial phases of oscillators
    Nt = obj.pNt;
    Nr = obj.pNr;
    NP = obj.pNumPaths;
    N1 = obj.NumSinusoids;

    w = generateRand(obj, 2*N1, NP*Nt*Nr);
    obj.pSinusoidsPhases = 2 * pi * reshape(w, 2*N1, 1, NP, Nt, Nr);    
  end  
  
  function updateInterpFilters(obj)
    % Reset time counter
    obj.pTimeCounter = 0;

    if obj.pHasPolyphaseFilter 
        % Initialize interpolation filter phase and indices
        initInterpFilters(obj);            
        % Estimate interpolation filter phases and indices based on current
        % time offset
        evolveInterpFilters(obj);
        % Reset polyphase filters
        resetPolyphaseFilters(obj);
        % Flush filters with one sample
        generatePathGains(obj, 1);
    else
        % This SOS model is rerun for extra samples equal to the polyphase
        % filter length. This is to ensure that the original fading samples
        % in stepImp are calculated at the right time instance.
        SOSGenerateOutput(obj, obj.pPolyphaseFilterLength);
    end
  end
    
  function evolveInterpFilters(obj)
    % Estimate interpolation filter phases and indices based on current
    % time offset
    TsSOS = obj.pSampleTime;
    initTime = obj.pInitialTime;
    
    isInitTimeOnSamplePt = (mod(initTime, TsSOS) == 0);

    % Estimate relative SOS sampling instance
    if ~isInitTimeOnSamplePt
        timeOffset = TsSOS * floor(initTime/TsSOS); 
    else
        timeOffset = initTime;
    end
    
    % Estimate relative channel sample position
    samplePosition = round((initTime - timeOffset)*obj.pSampleRate); % 0 or 1    
     
    % Estimate relative interpolation phases
    if samplePosition ~= 0
        estimateInterpFilterPhase(obj, samplePosition)
    end
    
    if ~isInitTimeOnSamplePt && (samplePosition ~= 1)
        % This is the special case, if samplePosition = 1 then handle
        % it similar to samplePosition=0.
        obj.pInitialTime = timeOffset + TsSOS;  
    else
        obj.pInitialTime = timeOffset;
    end
  end  
    
  function updateRicianLOSFhase(obj, deltaT)
    Rs = obj.pSampleRate;
    N = round(deltaT * Rs); 
    if N > 0 % Update starting LOS phase for the Rician channel
        obj.pLastThetaLOS = obj.pLastThetaLOS + N * cast( ...
            obj.DirectPathDopplerShift/Rs, obj.pInputDataType);
    end
  end 
  function updateNakagamiLOSFhase(obj, deltaT)
    Rs = obj.pSampleRate;
    N = round(deltaT * Rs); 
    if N > 0 % Update starting LOS phase for the Rician channel
        obj.pLastThetaLOS = obj.pLastThetaLOS + N * cast( ...
            obj.DirectPathDopplerShift/Rs, obj.pInputDataType);
    end
  end
end  

methods(Access = private) % Interpolation filter related methods
  function setupInterpFilters(obj, KI)
    % Set FNG/SOS handle
    if obj.pIsFGN
        obj.pFadingTechniqueHandle = @FGNGenerateOutput;
    else
        obj.pFadingTechniqueHandle = @SOSGenerateOutput;
    end    
    
    % Set up polyphase filter if necessary
    obj.pHasPolyphaseFilter = (KI(2) > 1);
    if obj.pHasPolyphaseFilter 
        inDT = obj.pInputDataType;
        M = obj.pNumPaths * obj.pNumLinks;
        obj.pIFilterInterpFactor = KI(2);
        L = obj.pPolyphaseFilterLength;
        b = intfilt(KI(2), L/2, 0.5);
        obj.pIFilterBank = cast(reshape([b 0], KI(2), L), inDT);
        % The following initialization is necessary for SOS burst mode
        obj.pIFilterPhase = KI(2) - (KI(3) <= 1);
        obj.pIFilterState = coder.nullcopy(complex( ...
            zeros(L-1, M*KI(2), inDT))); % [L-1, M*R]
        obj.pIFilterNewSampLastOut = coder.nullcopy(complex( ...
            zeros(KI(2), M, inDT))); % [R, M]
        obj.pIFilterLastOutputs = coder.nullcopy(complex( ...
            zeros(2, M, inDT))); % [2, M]
    end

    % Set up linear filter if necessary
    obj.pHasLinearFilter = (KI(3) > 1);
    if obj.pHasLinearFilter 
        obj.pLinearInterpFactor = KI(3);   
        % The following initialization is necessary for SOS burst mode
        obj.pLinearInterpIndex = 1;
    end          
  end
  
  function initInterpFilters(obj)
    % Initialize polyphase filter phase and linear filter index
    obj.pIFilterPhase = obj.pIFilterInterpFactor - (~obj.pHasLinearFilter);

    if obj.pHasLinearFilter
        obj.pLinearInterpIndex = 1;
    end    
  end
  
  function estimateInterpFilterPhase(obj, sampPosition)
    % Bring polyphase filter phase to linear filter index to the current
    % burst time for SOS
    IIF = obj.pIFilterInterpFactor;
    
    if ~obj.pHasLinearFilter
        % Increment polyphase filter phase by 1
        startPhase = mod(obj.pIFilterPhase, IIF) + 1;
        % Update polyphase filter phase
        obj.pIFilterPhase = mod(startPhase + (sampPosition-1) - 1, IIF) + 1;
    else
        LIF = obj.pLinearInterpFactor;
        % Linear interpolation indices.
        startIdx = obj.pLinearInterpIndex;
        endIdx = startIdx + sampPosition - 1;
        numSamples = ceil((endIdx-1) / LIF) + 1;
        if startIdx <= 2
            numNewOutputs = numSamples - 1;
        else
            numNewOutputs = numSamples - 2;
        end

        if (numNewOutputs > 0) && (numNewOutputs <= 1 + ceil(sampPosition/LIF))
            % Increment polyphase filter phase by 1
            startPhase = mod(obj.pIFilterPhase, IIF) + 1;
            % Update polyphase filter phase
            obj.pIFilterPhase = mod(startPhase + (numNewOutputs-1) - 1, IIF) + 1;
        end
        % Starting interpolation index for the next frame
        obj.pLinearInterpIndex = rem(endIdx, LIF) + 1;
    end
  end
  
  function resetPolyphaseFilters(obj)
    % Reset polyphase filter states given its phase and linear filter index
    M = obj.pNumPaths * obj.pNumLinks;        
    R = obj.pIFilterInterpFactor;
    L = obj.pPolyphaseFilterLength;
    x = obj.pFadingTechniqueHandle(obj,L);  %[L, M]
    inDT = obj.pInputDataType;

    obj.pIFilterState = complex(zeros(L-1, M*R, inDT)); % [L-1, M*R]
    filterOut = coder.nullcopy(complex(zeros(L*R, M, inDT)));
    for i = coder.unroll(1:R)
        [filterOut(i:R:(L-1)*R+i,:), obj.pIFilterState(:, (i-1)*M+(1:M))] = ...
            filter(obj.pIFilterBank(i,:), 1, x, zeros(L-1, M, inDT), 1);
    end
    obj.pIFilterNewSampLastOut = filterOut(end-R+(1:R), :); % [R, M];

    if obj.pIsFGN
        obj.pIFilterLastOutputs = [zeros(1, M, inDT); ...
            filterOut((L-1)*R+obj.pIFilterPhase,:)];
    else % SOS
        obj.pIFilterLastOutputs = filterOut(((L-1)*R+obj.pIFilterPhase-[1 0]),:);
    end
  end
  
  function y = generatePathGains(obj, N)
    if ~obj.pHasLinearFilter % No linear interpolation filters
        y = polyphaseFilterGenerateOutput(obj, N);
    else % Hybrid of polyphase filtering and linear interpolation
        M = obj.pNumPaths * obj.pNumLinks;
        LIF = obj.pLinearInterpFactor;
        inDT = obj.pInputDataType;

        % Linear interpolation indices
        startIdx = obj.pLinearInterpIndex;
        endIdx = startIdx + N - 1;

        % Number of samples required for linear interpolation. 
        % Special cases:
        % startIdx==1 and N==1 ==> numSamples = 1 (no interpolation)
        % startIdx==1 and N<=R+1 OR
        % startIdx==2 and N<=R ==> numSamples = 2 (2-pt interpolation)
        numSamples = ceil((endIdx-1) / LIF) + 1;

        % Determine *previous* polyphase filter outputs. If startIdx of
        % linear interpolation is 1 or 2, use only last filter output;
        % Otherwise, use last *two* outputs.
        if startIdx <= 2
            prevOutputs = obj.pIFilterLastOutputs(end, :);
            numNewOutputs = numSamples - 1;
        else
            prevOutputs = obj.pIFilterLastOutputs(end-[1 0], :);
            numNewOutputs = numSamples - 2;
        end

        % Samples for linear interpolation, [numSamples, M]. 
        % The upper bound on numNewOutputs is always met and unnecessary. 
        % Have it here for the object to work in MATLAB Function and MATLAB
        % System blocks (Accelerator mode).
        if (numNewOutputs > 0) && (numNewOutputs <= 1 + ceil(N/LIF))
            % Generate new polyphase filter outputs
            newOutputs = polyphaseFilterGenerateOutput(obj, numNewOutputs);
            linearInterpSamples = [prevOutputs; newOutputs]; 
        else
            linearInterpSamples = prevOutputs; 
        end

        % Perform linear interpolation on polyphase filter outputs
        D = [diff(linearInterpSamples, 1, 1); zeros(1, M, inDT)];
        b = cast((obj.pLinearInterpIndex - 1 + (0:N-1).')/LIF, inDT); 
        k = floor(b);
        y = linearInterpSamples(k+1,:) + bsxfun(@times, b-k, D(k+1, :));
        % The following code does the same linear interpolation using
        % the interp1 function, but tends to be slower. 
        % b = cast((obj.pLinearInterpIndex - 1 + (0:N-1).')/LIF, inDT); 
        % if 1 == size(linearInterpSamples,1)
        %    y = linearInterpSamples;
        % else
        %    y = interp1(0:size(linearInterpSamples,1)-1, linearInterpSamples, b); 
        % end

        % Starting interpolation index for the next frame
        obj.pLinearInterpIndex = rem(endIdx, LIF) + 1;            
    end
  end
  
  function y = polyphaseFilterGenerateOutput(obj, N)
    M = obj.pNumPaths * obj.pNumLinks;
       
    if ~obj.pHasPolyphaseFilter
        y = obj.pFadingTechniqueHandle(obj, N);  % Generate source samples.
    else
        R  = obj.pIFilterInterpFactor;
        FB = obj.pIFilterBank; % [R, L]
        inDT = obj.pInputDataType;

        % Initialize output
        y = coder.nullcopy(complex(zeros(N, M, inDT))); 

        % Increment polyphase filter phase by 1. 
        startPhase = mod(obj.pIFilterPhase, R) + 1; 

        % m0 is the number of samples to output *without* generating new
        % source samples. This is necessary when the starting phase is not
        % equal to 1.
        m0 = min(mod(R - (startPhase - 1), R), N);

        % Flush polyphase filter if needed.
        if m0 > 0
            y(1:m0, :) = obj.pIFilterNewSampLastOut((1:m0) + startPhase - 1, :);
        end

        % Required number of new source samples
        numNewSamps = ceil((N - m0)/R);        
        
        % Generate new source samples if necessary.        
        % The upper bound on numNewSamps is always met and unnecessary.
        % Have it here for the object to work in MATLAB Function and MATLAB
        % System blocks (Accelerator mode).
        if (numNewSamps > 0) && (numNewSamps <= ceil(N/R))
            x = obj.pFadingTechniqueHandle(obj, numNewSamps);
            filterOut = coder.nullcopy(complex(zeros(numNewSamps*R, M, inDT)));
            filterState = obj.pIFilterState; % [L-1, M*R]
            for i = coder.unroll(1:R)
                colIdx = (i-1)*M+(1:M);
                [filterOut(i:R:(numNewSamps-1)*R+i,:), filterState(:,colIdx)] = ...
                    filter(FB(i,:), 1, x, filterState(:,colIdx), 1); 
            end
            obj.pIFilterState = filterState;
            y(m0+1:N,:) = filterOut(1:N-m0,:);
            obj.pIFilterNewSampLastOut = filterOut(end-R+(1:R),:); % [R, M]
        end

        % Update polyphase filter phase
        obj.pIFilterPhase = mod(startPhase + (N-1) - 1, R) + 1;
        
        % Store last two outputs for each path. 
        if N == 1
            obj.pIFilterLastOutputs = [obj.pIFilterLastOutputs(end, :); y(1, :)]; % [2, M]
        else
            obj.pIFilterLastOutputs = y(end - [1 0], :); % [2, M]
        end
    end
  end  
end

methods(Access = protected) % Channel filter related methods
  function setupChannelFilter(obj)
    % As this method can be called from the info method and before the
    % setupImpl for an unlocked object, we cannot assume obj.pSampleRate
    % has been correctly set. 
    if (getExecPlatformIndex(obj) ~= 1)
        Rs = obj.SampleRate;
    else
        Rs = obj.pSampleRate;
    end
    
    obj.cChannelFilter = comm.internal.channel.ChannelFilter( ...
        'SampleRate', Rs, ...
        'PathDelays', obj.PathDelays);
    
    if getExecPlatformIndex(obj) == 1 % Simulink block
        obj.pChannelFilterInfo = info(obj.cChannelFilter);
    end
  end    
end

methods(Access = private) % RNG related methods
  function setupRNG(obj)
    if ~strcmp(obj.RandomStream, 'Global stream') 
        if isempty(coder.target)   
            obj.pRNGStream = RandStream(obj.RandomNumGenerator, 'Seed', obj.Seed);
        else
            obj.pRNGStream = coder.internal.RandStream(obj.RandomNumGenerator, 'Seed', obj.Seed);
        end
    end    
  end
  
  function resetRNG(obj)
    % Reset random number generator if it is not global stream    
    if ~strcmp(obj.RandomStream, 'Global stream') 
        reset(obj.pRNGStream, obj.Seed);
    end
  end
  
  function y = generateRandn(obj, numRows, numCols) 
    % Generate Gaussian distributed random numbers row-wisely    

    if strcmp(obj.RandomStream, 'Global stream')
        y = (randn(numCols, numRows, obj.pInputDataType)).'; 
    else
        y = (randn(obj.pRNGStream, numCols, numRows, obj.pInputDataType)).';
    end
  end
  
  function y = generateRand(obj, numRows, numCols) 
    % Generate uniform distributed random numbers row-wisely
    
    if strcmp(obj.RandomStream, 'Global stream') 
        y = (rand(numCols, numRows, obj.pInputDataType)).'; 
    else
        y = (rand(obj.pRNGStream, numCols, numRows, obj.pInputDataType)).'; 
    end
  end
end

methods(Access = private) % Visualization related methods
  function setupVisualization(obj)
    if ~obj.pHasPathGainVis && ~obj.pHasDopplerVis
        if ~isempty(obj.pPathGainVis)
            hide(obj.pPathGainVis);
        end
        if ~isempty(obj.pDopplerVis)
            hide(obj.pDopplerVis);
        end        
    else
        % Find column indices in 1:M (M = Nr x Nt x NP) that are specified for visualization
        MIdxIn3D = reshape(1:obj.pNumLinks*obj.pNumPaths, [obj.pNr, obj.pNt, obj.pNumPaths]);
        displayMIdx = MIdxIn3D(obj.AntennaPairsToDisplay(2), obj.AntennaPairsToDisplay(1),:);
        obj.pDisplayMIdx = displayMIdx(:);
        Rs = obj.pSampleRate;            
        
        if obj.pHasPathGainVis
            if isempty(obj.pPathGainVis)
                % Do not recreate one if it already exists, to avoid
                % time-consuming scope load time.
                obj.pPathGainVis = comm.internal.PathGainsVisualization;
            end
            
            chanFilterCoeff = obj.info.ChannelFilterCoefficients;
            tapDelays = -obj.info.ChannelFilterDelay + ...
                (0:size(chanFilterCoeff, 2) - 1);
            set(obj.pPathGainVis, ...
                'Display',             obj.Visualization, ...
                'SampleRate',          Rs, ...
                'PathDelays',          obj.PathDelays, ...
                'TapDelays',           tapDelays, ...
                'InterpolationMatrix', obj.info.ChannelFilterCoefficients, ...
                'NormalizedPathGains', obj.NormalizePathGains, ...
                'SamplesToDisplay',    obj.SamplesToDisplay);
            
            if ~obj.NormalizePathGains
                obj.pPathGainVis.MaximumAveragePathGain = max(obj.AveragePathGains);
            end

            if ~isempty(obj.pDopplerVis)
                hide(obj.pDopplerVis);
            end
        else % Doppler spectrum
            if isempty(obj.pDopplerVis)
                % Do not recreate one if it already exists, to avoid
                % time-consuming scope load time.
                obj.pDopplerVis = comm.internal.DopplerVisualization;
            end
            
            if obj.pIsStaticChannel 
                set(obj.pDopplerVis, 'StaticChannel', true);
            else 
                % Product of polyphase filter phase and linear filter interpolation factor
                interpFactor = 1;
                if obj.pHasPolyphaseFilter 
                    interpFactor = interpFactor*obj.pIFilterInterpFactor;
                end
                if obj.pHasLinearFilter 
                    interpFactor = interpFactor*obj.pLinearInterpFactor;
                end

                % Gaussian filter impulse response for the specified path
                if size(obj.pGFilterImpulseResponse, 1) == 1
                    IR = obj.pGFilterImpulseResponse;
                else
                    IR = obj.pGFilterImpulseResponse(obj.PathsForDopplerDisplay, :);
                end

                % Cutoff frequency for the specified path
                if length(obj.pFcFactor) == 1
                    fc = obj.MaximumDopplerShift.*obj.pFcFactor;
                else
                    fc = obj.MaximumDopplerShift.*obj.pFcFactor(obj.PathsForDopplerDisplay);
                end
                
                set(obj.pDopplerVis, ...
                    'StaticChannel',    false, ...
                    'SampleRate',       Rs/interpFactor, ...
                    'ImpulseResponse',  double(IR), ...
                    'CutoffFrequency',  fc, ...
                    'UpsamplingFactor', interpFactor);
            end

            if ~isempty(obj.pPathGainVis)
                hide(obj.pPathGainVis);
            end
        end
    end
  end
  
  function channelDisplay(obj, z)
    if obj.pHasPathGainVis && ~isAllScopesClosed(obj.pPathGainVis)
        % Update impulse and/or frequency response visualization
        step(obj.pPathGainVis, double(z(:, obj.pDisplayMIdx)));
    elseif obj.pHasDopplerVis && ~obj.pIsStaticChannel && ~isScopeClosed(obj.pDopplerVis)
        % Update text on dynamic channel Doppler spectrum display. We know
        % the accurate number of samples to the next update here, but not
        % at the step function call on obj.pDopplerVis, which only is
        % triggered when new Gaussian noise is generated. 
        setScopeText(obj.pDopplerVis, obj.pNumSamplesProcessed); 
    end
  end  
end

methods(Static, Hidden)
  function IR = getGaussianFilterIR(fgTs, fc, t, DS)
    % Set up the impulse response of the Gaussian filter
    
    fcLen = length(fc);
    IR = zeros(fcLen, length(t));

    for idx = 1:fcLen
        if isa(DS, 'doppler.baseclass') 
            curPathDS = DS(idx);
        elseif isa(DS, 'cell') 
            curPathDS = DS{idx};
        else
            curPathDS = DS;
        end    

        Noversampling = 1/fgTs/fc(idx)/2;

        switch curPathDS.SpectrumType
        case 'Jakes'
            IR(idx, :) = jakesir(fc(idx), t); 
        case 'Flat'
            IR(idx, :) = flatir(fc(idx), t);
        case 'Rounded'
            if isa(DS, 'doppler.baseclass') 
                IR(idx, :) = roundedir(fc(idx), curPathDS.CoeffRounded, Noversampling, t);
            else
                IR(idx, :) = roundedir(fc(idx), curPathDS.Polynomial, Noversampling, t);
            end
        case 'Bell'
            if isa(DS, 'doppler.baseclass') 
                IR(idx, :) = bellir(fc(idx), curPathDS.CoeffBell, Noversampling, t);
            else
                IR(idx, :) = bellir(fc(idx), curPathDS.Coefficient, Noversampling, t);
            end
        case 'RJakes'
            IR(idx, :) = rjakesir(fc(idx), curPathDS.FreqMinMaxRJakes, Noversampling, t);
        case 'Restricted Jakes'
            IR(idx, :) = rjakesir(fc(idx), curPathDS.NormalizedFrequencyInterval, Noversampling, t);
        case 'AJakes'        
            IR(idx, :) = ajakesir(fc(idx), curPathDS.FreqMinMaxAJakes, Noversampling, t);
        case 'Asymmetric Jakes'
            IR(idx, :) = ajakesir(fc(idx), curPathDS.NormalizedFrequencyInterval, Noversampling, t);
        case 'Gaussian'
            IR(idx, :) = gaussianir(fc(idx), t);
        case 'BiGaussian'
            if isa(DS, 'doppler.baseclass')         
                IR(idx, :) = bigaussianir(fc(idx), ...
                                          curPathDS.SigmaGaussian1, ...
                                          curPathDS.SigmaGaussian2, ...
                                          curPathDS.CenterFreqGaussian1, ...
                                          curPathDS.CenterFreqGaussian2, ...
                                          curPathDS.GainGaussian1, ...
                                          curPathDS.GainGaussian2, ...
                                          Noversampling, t);     
            else
                IR(idx, :) = bigaussianir(fc(idx), ...
                              curPathDS.NormalizedStandardDeviations(1), ...
                              curPathDS.NormalizedStandardDeviations(2), ...
                              curPathDS.NormalizedCenterFrequencies(1), ...
                              curPathDS.NormalizedCenterFrequencies(2), ...
                              curPathDS.PowerGains(1), ...
                              curPathDS.PowerGains(2), ...
                              Noversampling, t);            
            end
        end    
    end
  end    
  
  function interpMatrix = designInterpFilter(L, filterLen, phaseIdx, Astop)   
    interpMatrix = zeros(length(phaseIdx), filterLen);
    for i = 1:length(phaseIdx)
        if any(phaseIdx(i) == [1 L+1])
            interpMatrix(i, filterLen/2) = 1; % Not actually used
        else
            polyphaseFIR = designMultirateFIR(L, 1, filterLen/2, Astop(phaseIdx(i)));
            b = reshape(polyphaseFIR, L, []);
            interpMatrix(i, :) = b(phaseIdx(i), :);
        end
    end
  end
end

end

% [EOF]
