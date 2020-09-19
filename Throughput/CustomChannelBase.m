classdef (Hidden) CustomChannelBase < FadingChannel
%CustomChannelBase Base object to declare properties and share methods for
% a SISO fading channel

% Copyright 2017-2019 The MathWorks, Inc.

%#codegen

properties (Nontunable)
    %SampleRate Sample rate (Hz)
    %   Specify the sample rate of the input signal in Hz as a double
    %   precision, real, positive scalar. The default value of this
    %   property is 1 Hz.
    SampleRate = 1;
    %PathDelays Discrete path delays (s)
    %   Specify the delays of the discrete paths in seconds as a double
    %   precision, real, scalar or row vector. When PathDelays is a scalar,
    %   the channel is frequency-flat; When PathDelays is a vector, the
    %   channel is frequency-selective. The default value of this property
    %   is 0.
    PathDelays = 0;
    %AveragePathGains Average path gains (dB)
    %   Specify the average gains of the discrete paths in dB as a double
    %   precision, real, scalar or row vector. AveragePathGains must have
    %   the same size as PathDelays. The default value of this property is
    %   0.
    AveragePathGains = 0;
    %MaximumDopplerShift Maximum Doppler shift (Hz)
    %   Specify the maximum Doppler shift for the path(s) of the channel in
    %   Hz as a double precision, real, nonnegative scalar. It applies to
    %   all the paths of the channel. When MaximumDopplerShift is 0, the
    %   channel is static for the entire input and you can use the reset
    %   method to generate a new channel realization. The
    %   MaximumDopplerShift must be smaller than SampleRate/10/fc for each
    %   path, where fc is the cutoff frequency factor of the path. For a
    %   Doppler spectrum type other than Gaussian and BiGaussian, the value
    %   of fc is 1; Otherwise, the value of fc is dependent on the Doppler
    %   spectrum structure fields. Refer to the documentation of this
    %   System object for more details about how fc is defined. The default
    %   value of this property is 0.001.
    MaximumDopplerShift = 1e-3;
    %FadingTechnique Technique for generating fading samples
    %   Specify the fading technique as one of 'Filtered Gaussian noise' |
    %   'Sum of sinusoids'. The default value of this property is
    %   'Filtered Gaussian noise'.
    FadingTechnique = 'Filtered Gaussian noise';
    %NumSinusoids Number of sinusoids
    %   Specify the number of sinusoids used to generate the fading
    %   samples. This property applies when you set the FadingTechnique
    %   property to 'Sum of sinusoids'. The default value of this property
    %   is 48.
    NumSinusoids = 48;
    %InitialTimeSource Initial time source
    %   Specify the initial time source as one of 'Property'|'Input port'.
    %   This property applies when you set the FadingTechnique property to
    %   'Sum of sinusoids'. When 'InitialTimeSource' is set to 'Property',
    %   the 'InitialTime' property is used to specify the start time of the
    %   fading process. When 'InitialTimeSource' is set to 'Input port',
    %   the start time of the fading process is specified using the
    %   'INITIALTIME' input when the object is run. The input value can
    %   change when the object is run several times in a row.  The default
    %   value of this parameter is 'Property'.
    InitialTimeSource= 'Property';
    %InitialTime Initial time (s)
    %   Specify the start time for sum of sinusoids fading technique. 
    %   This property applies when the 'InitialTimeSource' parameter is set
    %   to 'Property'. The default value of this property is 0 seconds.
    InitialTime = 0;
    %Visualization Channel visualization 
    %   Specify which channel characteristic(s) to be visualized as one of
    %   'Off' | 'Impulse response' | 'Frequency response' | 'Impulse and
    %   frequency responses' | 'Doppler spectrum'. When visualization is
    %   on, separate window(s) show up to display the selected channel
    %   characteristic(s) such as impulse response or Doppler spectrum. The
    %   default value of this property is 'Off'.
    Visualization = 'Off';
    %PathsForDopplerDisplay Path for Doppler spectrum display
    %   Specify the discrete path for Doppler spectrum visualization as a
    %   numeric, real, positive integer scalar no larger than the length of
    %   PathDelays. This property applies when you set the Visualization
    %   property to 'Doppler spectrum'. The default value of this property
    %   is 1.
    PathsForDopplerDisplay = 1;
    %SamplesToDisplay Percentage of samples to display
    %   Specify the percentage of input samples, for which the channel
    %   characteristic(s) are displayed, as one of '10%' | '25%' | '50%' |
    %   '100%'. This property applies when you set the Visualization
    %   property to 'Impulse response', 'Frequency response' or 'Impulse
    %   and frequency responses'. The channel response(s) for a downsampled
    %   version of the input signal are visualized when this property is
    %   set to '10%', '25%' or '50%' for which the downsample factor is 10,
    %   4 and 2 respectively. The default value of this property is '25%'.
    SamplesToDisplay = '25%'    
end

properties(Constant, Hidden)
    FadingTechniqueSet = matlab.system.StringSet({ ...
        'Filtered Gaussian noise','Sum of sinusoids'});
    InitialTimeSourceSet = matlab.system.StringSet({ ...
        'Property','Input port'});    
    VisualizationSet = matlab.system.StringSet({ ...
        'Off', 'Impulse response', 'Frequency response', ...
        'Doppler spectrum', 'Impulse and frequency responses'});
    SamplesToDisplaySet = matlab.system.StringSet({ ...
        '10%','25%','50%','100%'});
end

methods
  function obj = CustomChannelBase(varargin) % Constructor
    coder.allowpcode('plain');
    setProperties(obj, nargin, varargin{:});
  end  
    
  function set.NumSinusoids(obj,nTerms)
    propName = 'NumSinusoids';
    validateattributes(nTerms, {'numeric'}, ...
        {'scalar','integer','>=',1}, ...
        [class(obj) '.' propName], propName);
    obj.NumSinusoids = nTerms;
  end
        
  function set.InitialTime(obj,initialTime)
    propName = 'InitialTime';
    validateattributes(initialTime, {'double'}, ...
        {'real','>=',0,'scalar','finite'}, ...
        [class(obj) '.' propName], propName);
    obj.InitialTime = initialTime;
  end
  
  function set.SampleRate(obj, Rs)
    propName = 'SampleRate';
    validateattributes(Rs, {'double'}, ...
        {'real','scalar','positive','finite'}, ...
        [class(obj) '.' propName], propName); 

    obj.SampleRate = Rs;
  end
  
  function set.PathDelays(obj, tau)
    propName = 'PathDelays';
    validateattributes(tau, {'double'}, {'real','row','finite'}, ...
        [class(obj) '.' propName], propName); 

    obj.PathDelays = tau;
  end
  
  function set.AveragePathGains(obj, PdB)
    propName = 'AveragePathGains';
    validateattributes(PdB, {'double'}, {'real','row','finite'}, ...
        [class(obj) '.' propName], propName); 

    obj.AveragePathGains = PdB;
  end

  function set.MaximumDopplerShift(obj, fd)
    propName = 'MaximumDopplerShift';
    validateattributes(fd, {'double'}, ...
        {'real','scalar','nonnegative','finite'}, ...
        [class(obj) '.' propName], propName); 

    obj.MaximumDopplerShift = fd;
  end   
  
  function set.PathsForDopplerDisplay(obj, dispPath)
    propName = 'PathsForDopplerDisplay';
    validateattributes(dispPath, {'numeric'}, ...
        {'real','scalar','positive','integer','finite'}, ...
        [class(obj) '.' propName], propName);

    obj.PathsForDopplerDisplay = dispPath;
  end
end

methods(Access = protected)   
  function validatePropertiesImpl(obj)
    % Check AveragePathGains and DopplerSpectrum sizes
    validateDelayProfileAndDoppler(obj);
    
    % Doppler spectra have to be Jakes for SOS
    if strcmp(obj.FadingTechnique, 'Sum of sinusoids')
        validateDopplerSpectrumSOS(obj);
    end
    
    % No visualization for SOS modeling
    coder.internal.errorIf(~strcmp(obj.Visualization, 'Off') && ...
        strcmp(obj.FadingTechnique, 'Sum of sinusoids'), ...
        'comm:FadingChannel:NoVisualizationForSOS');            

    % No code generation when visualization is on
    coder.internal.errorIf(~isempty(coder.target) && ...
        ~strcmp(obj.Visualization, 'Off'), ...
        'comm:FadingChannel:NoCodegenForVisual');    
  end
  
  function validateNakagamiProperties(obj)
    NP = length(obj.PathDelays);
    K = obj.KFactor;
    m = obj.mvalue;
    
    coder.internal.errorIf(~isscalar(K) && ~isequal(length(K), NP), ...
        'comm:FadingChannel:UnequalKFactorTauLen');

    coder.internal.errorIf(~isequal(size(obj.DirectPathDopplerShift), size(K)), ...
        'comm:FadingChannel:UnequalLOSShiftKFactorLen');

    if (obj.MaximumDopplerShift == 0) && any(obj.DirectPathDopplerShift > 0)
        coder.internal.warning('comm:FadingChannel:LOSShiftForStaticChan');
    end

    coder.internal.errorIf(~isequal(size(obj.DirectPathInitialPhase), size(K)), ...
        'comm:FadingChannel:UnequalLOSPhaseKFactorLen');

    if (obj.MaximumDopplerShift == 0) && any(obj.DirectPathInitialPhase > 0)
        coder.internal.warning('comm:FadingChannel:LOSPhaseForStaticChan');
    end      
  end
  
  
  function s = infoImpl(obj)
    %info Returns characteristic information about the channel
    %   S = info(OBJ) returns a structure containing characteristic
    %   information, S, about the fading channel. A description of the
    %   fields and their values is as follows:
    % 
    %   ChannelFilterDelay        - Channel filter delay (samples)
    %   ChannelFilterCoefficients - Coefficient matrix used to convert path
    %                               gains to channel filter tap gains for
    %                               each sample and each pair of transmit
    %                               and receive antennas. 
    %   NumSamplesProcessed       - Number of samples the channel has 
    %                               processed since the last reset
    %   LastFrameTime             - Last frame ending time (seconds). This
    %                               field applies when the FadingTechnique
    %                               property is set to 'Sum of sinusoids'
    %                               and the InitialTimeSource property is
    %                               set to 'Input port'.
    
    if ~isempty(coder.target) || ~isLocked(obj)
        % Cannot define cChannelFilter here due to codegen limitation. Have
        % to create a temporary channel filter object to work around this. 
        chanFilterObj = comm.internal.channel.ChannelFilter( ...
            'SampleRate', obj.SampleRate, ...
            'PathDelays', obj.PathDelays);        
        s = info(chanFilterObj);
    else
        s = info(obj.cChannelFilter);
    end
    
    s.NumSamplesProcessed = obj.pNumSamplesProcessed * (isLocked(obj));
    if (strcmp(obj.FadingTechnique,'Sum of sinusoids') && ...
        strcmp(obj.InitialTimeSource,'Input port'))
        if ~isLocked(obj)
            s.LastFrameTime = 0;
        else
            s.LastFrameTime = obj.pLastFrameEndTime - ...
                double(obj.pLastFrameEndTime > 0)/obj.SampleRate;
        end
    end
  end  
end

methods(Access = private)   
  function validateDelayProfileAndDoppler(obj)
    % Check AveragePathGains size
    NP = length(obj.PathDelays);
    coder.internal.errorIf(~isequal(length(obj.AveragePathGains), NP), ...
        'comm:FadingChannel:UnequalTauPdBLen');

    % Check DopplerSpectrum size
    DS = obj.DopplerSpectrum;
    coder.internal.errorIf(~isscalar(DS) && ~isequal(length(DS), NP), ...
        'comm:FadingChannel:UnequalDSTauLen');
  
    % Check PathsForDopplerDisplay value
    coder.internal.errorIf(strcmp(obj.Visualization, 'Doppler spectrum') && ...
        (obj.PathsForDopplerDisplay > NP), 'comm:FadingChannel:DopplerDispPathOutOfRange');
  end
    
  function validateDopplerSpectrumSOS(obj)
    if isa(obj.DopplerSpectrum, 'struct') % Single Doppler structure
         coder.internal.errorIf( ...
             ~strcmp(obj.DopplerSpectrum.SpectrumType,'Jakes'),...
            'comm:FadingChannel:DopplerSpectrumCheckSOS');
    elseif isa(obj.DopplerSpectrum, 'cell') % Cell array of Doppler struct
        coder.internal.errorIf(~all(strcmp('Jakes', ...
            cellfun(@(x)(x.SpectrumType), obj.DopplerSpectrum, ...
            'UniformOutput', false))), ...
            'comm:FadingChannel:DopplerSpectrumCheckSOS');
    else % Doppler object scalar or array
        coder.internal.errorIf(~isa(obj.DopplerSpectrum, 'doppler.jakes'), ...
            'comm:FadingChannel:DopplerSpectrumCheckSOS');        
    end
  end
end

methods(Static, Hidden)
  function fc = getCutoffFreqFactor(DS)
    % Set up cutoff frequency factor of each discrete path.

    fc = zeros(1, length(DS));

    for idx = 1 : length(DS)
        if isa(DS, 'doppler.baseclass') 
            curPathDS = DS(idx);
        elseif isa(DS, 'cell') 
            comm.internal.CustomChannelBase.validateDopplerStruct(DS{idx});
            curPathDS = DS{idx};
        else
            comm.internal.CustomChannelBase.validateDopplerStruct(DS);
            curPathDS = DS;
        end

        switch curPathDS.SpectrumType
        case 'Jakes'
            fc(idx) = 1.0;
        case 'Flat'
            fc(idx) = 1.0;
        case 'Rounded'
            fc(idx) = 1.0;
        case 'Bell'
            fc(idx) = 1.0;
        case 'RJakes'
            % Do error checking on FreqMinMaxRJakes property: need a minimum
            % frequency separation to ensure that the impulse response is
            % correctly computed.
            coder.internal.errorIf((diff(curPathDS.FreqMinMaxRJakes) <= 1/50), ...
                'comm:FadingChannel:FreqMinMaxRJakes');
            fc(idx) = 1.0;
        case 'Restricted Jakes'
            % Do error checking on NormalizedFrequencyInterval field
            coder.internal.errorIf((diff(curPathDS.NormalizedFrequencyInterval) <= 1/50), ...
                'comm:FadingChannel:FreqMinMaxRJakes');
            fc(idx) = 1.0;        
        case 'AJakes'
            % Do error checking on FreqMinMaxAJakes property: need a minimum
            % frequency separation to ensure that the impulse response is
            % correctly computed.
            coder.internal.errorIf((diff(curPathDS.FreqMinMaxAJakes) <= 1/50), ...
                'comm:FadingChannel:FreqMinMaxAJakes');
            fc(idx) = 1.0;
        case 'Asymmetric Jakes'
            % Do error checking on NormalizedFrequencyInterval field
            coder.internal.errorIf((diff(curPathDS.NormalizedFrequencyInterval) <= 1/50), ...
                'comm:FadingChannel:FreqMinMaxAJakes');
            fc(idx) = 1.0;        
        case 'Gaussian' 
            if isa(DS, 'doppler.baseclass') 
                fc(idx) = curPathDS.SigmaGaussian * sqrt(2*log(2));
            else
                fc(idx) = curPathDS.NormalizedStandardDeviation * sqrt(2*log(2));
            end
        case 'BiGaussian'
            if isa(DS, 'doppler.baseclass') 
                if ( (curPathDS.GainGaussian1 == 0) && (curPathDS.CenterFreqGaussian2 == 0) )
                    fc(idx) = curPathDS.SigmaGaussian2 * sqrt(2*log(2));
                elseif (curPathDS.GainGaussian2 == 0) && (curPathDS.CenterFreqGaussian1 == 0) 
                    fc(idx) = curPathDS.SigmaGaussian1 * sqrt(2*log(2));
                elseif ((curPathDS.CenterFreqGaussian1 == 0) && (curPathDS.CenterFreqGaussian2 == 0) && ...
                        (curPathDS.SigmaGaussian1 == curPathDS.SigmaGaussian2))
                    fc(idx) = curPathDS.SigmaGaussian1 * sqrt(2*log(2));
                else % "True" bi-Gaussian case
                    coder.internal.errorIf(...
                        (abs(curPathDS.CenterFreqGaussian1) + curPathDS.SigmaGaussian1*sqrt(2*log(2)) > 1), ...
                        'comm:MIMOChannel:CenterFreqGaussian1SigmaGaussian1');
                    coder.internal.errorIf(...
                        (abs(curPathDS.CenterFreqGaussian2) + curPathDS.SigmaGaussian2*sqrt(2*log(2)) > 1), ...
                        'comm:MIMOChannel:CenterFreqGaussian2SigmaGaussian2');
                    fc(idx) = 1.0;
                end
            else
                if ( (curPathDS.PowerGains(1) == 0) && (curPathDS.NormalizedCenterFrequencies(2) == 0) )
                    fc(idx) = curPathDS.NormalizedStandardDeviations(2) * sqrt(2*log(2));
                elseif (curPathDS.PowerGains(2) == 0) && (curPathDS.NormalizedCenterFrequencies(1) == 0) 
                    fc(idx) = curPathDS.NormalizedStandardDeviations(1) * sqrt(2*log(2));
                elseif (isequal(curPathDS.NormalizedCenterFrequencies, [0 0]) && ...
                        (diff(curPathDS.NormalizedStandardDeviations) == 0))
                    fc(idx) = curPathDS.NormalizedStandardDeviations(1) * sqrt(2*log(2));
                else % "True" bi-Gaussian case
                    coder.internal.errorIf(...
                        any(abs(curPathDS.NormalizedCenterFrequencies) + ...
                        curPathDS.NormalizedStandardDeviations*sqrt(2*log(2)) > 1), ...
                        'comm:FadingChannel:CenterFreqPowerGainMismatch');
                    fc(idx) = 1.0;
                end                        
            end
        end    
    end
  end

  function validateDopplerStruct(ds)
    % Verify input to be a valid Doppler structure
    
    coder.internal.errorIf(~isfield(ds, 'SpectrumType'), ...
        'comm:doppler:NoSpectrumTypeField');
    coder.internal.errorIf(~comm.internal.utilities.isCharOrStringScalar(ds.SpectrumType), ...
        'comm:doppler:InvalidSpecType');

    numFields = length(fields(ds));

    switch ds.SpectrumType
    case 'Jakes'
        coder.internal.errorIf((numFields ~= 1), ...
            'comm:doppler:InvalidJakesField');
    case'Flat'
        coder.internal.errorIf((numFields ~= 1), ...
            'comm:doppler:InvalidFlatField');
    case 'Rounded'
        coder.internal.errorIf((numFields ~= 2) || ...
            ~isfield(ds, 'Polynomial'), 'comm:doppler:InvalidRoundedField');
        doppler('Rounded', ds.Polynomial);
    case 'Bell'
        coder.internal.errorIf((numFields ~= 2) || ...
            ~isfield(ds, 'Coefficient'), 'comm:doppler:InvalidBellField');
        doppler('Bell', ds.Coefficient);
    case 'Asymmetric Jakes'
        coder.internal.errorIf((numFields ~= 2) || ...
            ~isfield(ds, 'NormalizedFrequencyInterval'), 'comm:doppler:InvalidAJakesField');
        doppler('Asymmetric Jakes', ds.NormalizedFrequencyInterval);
    case 'Restricted Jakes'
        coder.internal.errorIf((numFields ~= 2) || ...
            ~isfield(ds, 'NormalizedFrequencyInterval'), 'comm:doppler:InvalidRJakesField');
        doppler('Restricted Jakes', ds.NormalizedFrequencyInterval);
    case 'Gaussian'
        coder.internal.errorIf((numFields ~= 2) || ...
            ~isfield(ds, 'NormalizedStandardDeviation'),  'comm:doppler:InvalidGaussianField');
        doppler('Gaussian', ds.NormalizedStandardDeviation);
    case 'BiGaussian'
        coder.internal.errorIf((numFields ~= 4) || ...
            ~isfield(ds, 'NormalizedStandardDeviations') || ...
            ~isfield(ds, 'NormalizedCenterFrequencies') || ...
            ~isfield(ds, 'PowerGains'), 'comm:doppler:InvalidBiGaussianFields');
        doppler('BiGaussian', ...
            'NormalizedStandardDeviations', ds.NormalizedStandardDeviations, ...
            'NormalizedCenterFrequencies',  ds.NormalizedCenterFrequencies, ...
            'PowerGains',                   ds.PowerGains);
    otherwise
        coder.internal.errorIf(true, 'comm:doppler:InvalidSpecType');  
    end
  end   
end

end

% [EOF]
