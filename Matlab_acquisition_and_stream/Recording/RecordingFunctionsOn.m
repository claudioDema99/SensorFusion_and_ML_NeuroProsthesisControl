%This is the second class among the four main class of this program
%This class will handle all the stuff related to start snd stop recording and connecting
%to devices, and disconnecting
classdef RecordingFunctionsOn < handle

    properties (Access = public)
        signal SignalOn;
        RecordingStat logical = 0;
        DrawnIdx=[];

        % === Statistics ===
        statTotalReceivedBytes=0; % on MAIN pipe
        % added features for tcp communication
        %tcpServer;
        %MainSrcSamples;
        %AuxSrcSamples;
        %i = 0;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % ROS 2 properties
        node;
        publisher_emg;
        publisher_imu;
        %topic = "/delsys_data";
    end

    properties (Access = private)
        isConnected logical = 0;

        recProps RecordingPropertiesOn;

        % Polling timer stuff
        pollTimer = timer.empty; %Timer object
        tStartRec = 0; %Starting time for recording
        tLastPoll = 0; %Last timer callback time

        streams MultiStreamSourcedSignal;

        % === dummy ===
        dummyLastSample = 0;
        dummyCurrentTime = 0;

        % === DELSYS ===
        objsendto = [];
        objreadfromEMG = [];
        objreadfromAUX = [];
        hasStreamAux logical = false;

        % === Pre-recorded Signal File ===
        prerecSignal SignalOn = SignalOn.empty;
        prerecCurrentTime = 0;
        prerecPlaybackSpeedRatio = 0; % For future use
        %prerecGui GUI_ReplayTool;
        prerecImportTrueTags logical = true;
        prerecTrueTagPtr = -1;
    end

    properties (Constant)
        SwingMark= 9;
        StanceMark= 8;
        SizeofDelsysEMG= 64; %16channels,  % Each sample is 64 bytes = 16channel x 4byte/sample.
        SizeofDelsysAUX= 576; %16channels  %  % Each sample is 576 bytes = 144 channel x 4byte/sample.
        MaxEMGChan=16;
        MaxAUXChan=144;
    end

    methods
        function this = RecordingFunctionsOn(rec_props)
            % Ctor
            this.recProps = rec_props;
            this.signal = SignalOn(rec_props);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %this.tcpServer = tcpserver('127.0.0.1', 51234);
            %configureCallback(this.tcpServer,"byte",3,@this.sendStoredData);
            % Create a ROS 2 node
            this.node = ros2node("/recording_node");
            % Create a publisher on the specified topic
            %this.publisher = ros2publisher(this.node, this.topic, "std_msgs/String");
            this.publisher_emg = ros2publisher(this.node, '/float64emg', 'std_msgs/Float64MultiArray');
            this.publisher_imu = ros2publisher(this.node, '/float64imu', 'std_msgs/Float64MultiArray');
        end

        function rp = GetRecProps(this)
            rp = this.recProps;
        end

        function t = GetPrerecordedPlaybackTime(this)
            if ~strcmp(this.recProps.Device, "prerecorded")
                error("Invalid call to GetPrerecordedPlaybackTime")
            end
            t = this.prerecCurrentTime;
        end

        function SetPrerecordedPlaybackTime(this, t)
            if ~strcmp(this.recProps.Device, "prerecorded")
                error("Invalid call to GetPrerecordedPlaybackTime")
            end
            if isnan(t)
                error("Invalid time specified for SetPrerecordedPlaybackTime()")
            end
            maxTime = this.GetPrerecordedSignalLength();
            if t > maxTime
                t = maxTime;
            end
            if t < 0
                t = 0;
            end
            this.prerecCurrentTime = t;
            this.prerecTrueTagPtr = -1; % invalidate true tag pointer.
        end

        function SetPrerecordedImportTrueTags(this, enable)
            if ~strcmp(this.recProps.Device, "prerecorded")
                error("Invalid call to SetPrerecordedImportTrueTags")
            end
            this.prerecImportTrueTags = enable;
            this.prerecTrueTagPtr = -1; % invalidate true tag pointer.
        end

        function enable = GetPrerecordedImportTrueTags(this)
            enable = this.prerecImportTrueTags;
        end

        function sig = GetPrerecordedSignal(this)
            if ~strcmp(this.recProps.Device, "prerecorded")
                error("Invalid call to GetPrerecordedSignal")
            end
            sig = this.prerecSignal;
        end

        function L = GetPrerecordedSignalLength(this)
            if ~strcmp(this.recProps.Device, "prerecorded")
                error("Invalid call to GetPrerecordedSignalLength")
            end

            % Note that when calculating length of source signal, the
            % original samplerate (sig.RecProps.SamplingFreq) doesn't
            % matter!
            sig = this.prerecSignal;
            L = sig.GetNumSample() / this.recProps.SamplingFreq;
        end

        function delete(this)
            % Make sure to stop and disconnect
            this.StopRecording()
            this.Disconnect()
        end

        function y = IsConnected(this)
            y = this.isConnected;
        end

        function y = IsRecordingStarted(this)
            y = this.isConnected && ~isempty(this.pollTimer);
        end

        function Disconnect(this)
            % !@! CLEANUP

            % First do a cleanup of recording
            this.StopRecording()

            % Cleanup
            this.statTotalReceivedBytes = 0;

            if ~isempty(this.recProps)
                % Disconnect from device
                if strcmp(this.recProps.Device, 'DELSYS_Trigno')
                    % Handle DELSYS
                    gprlog('Disconnecting from DELSYS');

                    if this.recProps.UnmaskedHasIMU==1
                        gprlog('Disconnecting from DELSYS_IMU');
                    end

                    % Close TCP/IP
                    if ~isempty(this.objsendto)
                        clear this.objsendto
                        this.objsendto = [];
                    end
                    if ~isempty(this.objreadfromEMG)
                        clear this.objreadfromEMG
                        this.objreadfromEMG = [];
                    end
                    if ~isempty(this.objreadfromAUX)
                        clear this.objreadfromAUX
                        this.objreadfromAUX = [];
                    end

                end

                this.isConnected = 0;
            end
        end
        function ConnectToDevice(this)
            if this.isConnected
                gprlog('* Already connected!')
                return
            end

            % Verify rec props. Would crash if any issues.
            this.recProps.Verify();

            % Connect
            deviceName  = this.recProps.Device;
            commType = this.recProps.ComType;
            nCh = this.recProps.UnmaskedNumEMGCh+this.recProps.UnmaskedNumPSCh;
            %nChAux = this.recProps.UnmaskedNumIMUCh;
            %sF = this.recProps.SamplingFreq;

            if nCh < 0
                gprlog('* Invalid number of channels')
                return
            end

            if strcmp(deviceName, 'DELSYS_Trigno') == 1 && strcmp(commType, 'WiFi') == 1
                try
                    if nCh > 16
                        gprlog('* DELSYS expects maximum channel count of 16');
                        return
                    end

                    % Open the connection to data port
                    this.objreadfromEMG = tcpclient('localhost',50043);
                    this.objreadfromEMG.ByteOrder = "little-endian";
                    %this.objreadfromEMG.InputBufferSize = 64000000;

                    % Open command port
                    this.objsendto = tcpclient('localhost',50040);
                    configureTerminator(this.objsendto, "CR/LF")
                    writeline(this.objsendto, sprintf('MASTER\r\nSTOP\r\nENDIAN LITTLE\r\n')); % Will add extra \r\n
                    pause(0.1);
                    flush(this.objsendto)

                    % Do we have IMU data for Delsys?
                    if this.recProps.UnmaskedHasIMU==1
                        this.hasStreamAux = true;
                    else
                        this.hasStreamAux = false;
                    end

                    if this.hasStreamAux
                        % Open the connection to data port for IMU
                        this.objreadfromAUX = tcpclient('localhost',50044);
                        this.objreadfromAUX.ByteOrder = "little-endian";
                        %this.objreadfromAUX.InputBufferSize = 64000000;
                        flush(this.objreadfromAUX)
                    end

                    % Read available data from data port and discard it
                    flush(this.objreadfromEMG)
                    gprlog('Connected to Delsys!')

                    % Generate streams handler for DELSYS.
                    % This is the signal configuration in DELSYS.
                    this.streams = MultiStreamSourcedSignal();
                    if this.hasStreamAux
                        this.streams.Initialize({'MAIN', 'AUX'}, [16 144], [2000 4000/27]); % 4000/27=148.148
                    else
                        this.streams.Initialize({'MAIN'}, 16, 2000);
                    end

                catch Ex
                    % Clean up
                    clear this.objreadfromEMG
                    clear this.objsendto
                    clear this.objreadfromAUX

                    this.objreadfromEMG = [];
                    this.objreadfromAUX = [];
                    this.objsendto = [];

                    gprlog('* CONNECTION ERROR: please start the DELSYS Trigno Control Utility or download it from: http://www.delsys.com/Attachments_pdf/installer/TrignoSDKServerInstaller.exe or from DELSYS website: http://www.delsys.com/integration/sdk/ - Internal error: %s', GetExceptionSummary(Ex));
                    rethrow(Ex);

                    %return
                end
            else
                gprlog('* Unknown device/comm type %s/%s', deviceName, commType);
                return
            end

            this.isConnected = 1;

        end

        function StartRecording(this)
            if ~this.isConnected
                gprlog('* StartRecording (RecSess): Device is not connected yet.')
                return
            end

            this.RecordingStat =1;
            deviceName_EMG  = this.recProps.Device;

            % Store timer start time
            this.tStartRec = tic;
            this.tLastPoll = this.tStartRec;

            % Clear signal log
            this.signal.ClearSignal();

            % Send start command
            if strcmp(deviceName_EMG, 'DELSYS_Trigno')
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Open connection to TCP server
                %fopen(this.tcpServer);
                
                % Handle DELSYS
                % Send the commands to start data streanng
                flush(this.objsendto);
                if this.hasStreamAux
                    flush(this.objreadfromAUX);
                end
                flush(this.objreadfromEMG);

                this.streams.ClearStreams();

                gprlog('Starting recording on DELSYS...');
                writeline(this.objsendto, sprintf('ENDIAN LITTLE\r\nSTART\r\n')); % will add extra \r\n
                pause(0.1);
            end

            % Create data polling timer
            this.pollTimer = timer('Name', 'PollTimer', ...
                'Period', 0.001, ...
                'StartDelay', 0, ...
                'TasksToExecute', inf, ...
                'ExecutionMode', 'fixedRate', ...
                'TimerFcn', @this.OnPollData, ...
                'ErrorFcn', @this.OnPollError);
            start(this.pollTimer);
            disp('Poll Timer Started')
        end

        function OnPollData(this, tmrObj, ~) %(this, tmrObj, eventArgs)
            % Check that this is a valid call to OnPollData
            try
                if tmrObj ~= this.pollTimer
                    throw(MException("LOCOD:InternalError", "Invalid timer object for OnPollData. This is an internal error."))
                end
                if ~this.isConnected
                    throw(MException("LOCOD:InternalError", "Device is not connected"))
                end
            catch Ex
                % Something is wrong
                stop(tmrObj);
                delete(tmrObj);
                this.pollTimer = timer.empty;

                gprlog('* Invalid call to data poll timer: %s', GetExceptionSummary(Ex));
                throw(MException("LOCOD:PollDataFailed", "Exception " + ...
                    "caught within onPollData. See console for more info."))
            end

            try
                % Read delta time
                t = tic;
                %deltaTime = double(t - this.tLastPoll) / 1.0e7; % 1.0e7 is resolution of tic()
                this.tLastPoll = t;
                if strcmp(this.recProps.Device, 'DELSYS_Trigno')
                    % === Handle DELSYS === %
                    % Delete the data inside the variables NOTSUREABOUTIT
                    %MainSrcSamples = [];
                    %AuxSrcSamples = [];

                    % Find out how many samples to read. Check available
                    % data count and read all available samples.
                    % MAIN pipe: Always 16 channels (64 B)
                    % AUX pipe : Always 9x16 = 144 channels (576 B)
                    MainSrcMaxSample = floor(this.objreadfromEMG.NumBytesAvailable / 64);
                    if this.hasStreamAux
                        AuxSrcMaxSample = floor(this.objreadfromAUX.NumBytesAvailable / 576);
                    else
                        AuxSrcMaxSample = 0;
                    end

                    % Read the MAIN pipe
                    if MainSrcMaxSample ~= 0
                        % Read MAIN data. Originally it is Float32 (Flt)
                        MainSrcSamplesVectorFlt = read(this.objreadfromEMG, MainSrcMaxSample * 11, 'single');
                        MainSrcSamplesVector = cast(MainSrcSamplesVectorFlt, 'double');
                        MainSrcSamples = reshape(MainSrcSamplesVector, [11 MainSrcMaxSample]);

                        this.statTotalReceivedBytes = this.statTotalReceivedBytes + this.objreadfromEMG.NumBytesAvailable;
                    else
                        MainSrcSamples = zeros(11, 0);
                    end

                    % Read the AUX pipe
                    if AuxSrcMaxSample ~= 0
                        AuxSrcSamplesVectorFlt = read(this.objreadfromAUX, AuxSrcMaxSample * 18, 'single');
                        AuxSrcSamplesVector = cast(AuxSrcSamplesVectorFlt, 'double');
                        AuxSrcSamples = reshape(AuxSrcSamplesVector, [18 AuxSrcMaxSample]);
                    else
                        AuxSrcSamples = zeros(18, 0);
                    end

                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % Convert data to string (for example)
                    %data_to_send = [MainSrcSamples; AuxSrcSamples];
                    % Create a ROS 2 message
                    msg_emg = ros2message(this.publisher_emg);
                    msg_imu = ros2message(this.publisher_imu);
                    if (size(MainSrcSamples, 1) == 11 && size(AuxSrcSamples, 1) == 1)%MainSrcSamples(:), 2) >= 1)
                        if (size(MainSrcSamples, 2) >= 50 && size(AuxSrcSamples, 2) >= 50)
                            % Determine indices for downsampling
                            indicesMain = round(linspace(1, size(MainSrcSamples, 2), 50));
                            indicesAux = round(linspace(1, size(AuxSrcSamples, 2), 50));
                            % Downsample by selecting the calculated indices
                            MainToSend = MainSrcSamples(:, indicesMain);
                            AuxToSend = AuxSrcSamples(:, indicesAux);
                            if (size(MainToSend, 2) == 50 && size(AuxToSend, 2) == 50)
                                msg_emg.data = MainToSend(:);
                                msg_imu.data = AuxToSend(:);
                                send(this.publisher_emg, msg_emg);
                                send(this.publisher_imu, msg_imu);
                            end
                        end
                    end
                    % If actually recording, go further.
                    if this.RecordingStat == 1 && size(MainSrcSamples, 2) > 0
                        this.IncomingDataTransposition_Delsys(MainSrcSamples, AuxSrcSamples);
                    end
                else
                    error('OnPollData: Invalid device')
                end

            catch Ex
                gprlog('* Exception in Poll Timer: \n%s', GetExceptionSummary(Ex));
                throw(MException("LOCOD:PollDataFailed", "Exception " + ...
                    "caught within onPollData. See console for more info."))
            end
        end

        function StopRecording(this)
            this.RecordingStat=0;

            % Close connection to TCP server
            %fclose(this.tcpServer);

            % !@! CLEANUP
            if ~isempty(this.recProps)
                % Handle DELSYS
                if strcmp(this.recProps.Device, 'DELSYS_Trigno')
                    if ~isempty(this.objsendto)
                        % Stop the aquisition
                        writeline(this.objsendto,  sprintf('STOP\r\n'));
                        pause(0.1)
                        flush(this.objsendto);
                    end
                end

                % Handle prerecorded
                % !@! WASBUGGY: Must not clear prerecSignal yet.
            end

            % Delete helper GUI for prerecorded
            %{
            if ~isempty(this.prerecGui)
                try
                    this.prerecGui.InvalidateAndExit();
                catch Ex
                    warning(['Error while deleting playback GUI: ' Ex.message])
                end
                this.prerecGui = GUI_ReplayTool.empty();
            end
            %}

            % Cleanup Streams manager
            this.streams = MultiStreamSourcedSignal.empty;

            % Kill the timer
            if ~isempty(this.pollTimer)
                gprlog("Stopping poll timer")
                stop(this.pollTimer);
                delete(this.pollTimer);
                this.pollTimer = [];
            end
        end

        function OnPollError(~, ~, eventArgs) %(this, tmrObj, eventArgs)
            gprlog(['* Error occured with while capturing data within timer ' ...
                '-- See console for more info: %s'], eventArgs.Data.message)
        end

        function IncomingDataTransposition_Delsys(this, MainSrcSamples, AuxSrcSamples)
            % Data from Delsys is configured in the following way:
            % MainSrc: Always 16 channels x NumSampleMain
            % AuxSrc: Always 144 channels x NumSampleAux
            %~~[~, nSmpMain] = size(MainSrcSamples);
            %~~[~, nSmpAux] = size(AuxSrcSamples);

            % Now! Sample rate at MAIN should be 2000 (see RecProps.Verify)
            % Sample rate at AUX should be 4000 / 27 = 148.148148...
            % The data at AUX should be upsampled to same sample rate of
            % MAIN. Store data inside MAIN
            % More info on Trigno SDK datasheet.
            % TODO: Check UnmaskedHasEMG
            if this.recProps.UnmaskedHasEMG==1
                EMGSig = MainSrcSamples(this.recProps.ChannelSelectionEMG, :);
            else
                EMGSig = [];
            end

            if this.recProps.UnmaskedHasIMU==1
                IMUSig = AuxSrcSamples(this.recProps.ChannelSelectionIMU, :);
            else
                IMUSig = [];
            end

            % Get the number of samples (columns) in each signal
            len_emg = size(EMGSig, 2);
            len_imu = size(IMUSig, 2);
            
            % Check lengths and truncate the longer signal
            if len_emg > len_imu
                % Truncate EMGSig to match the length of IMUSig
                EMGSig = EMGSig(:, 1:len_imu);
            elseif len_imu > len_emg
                % Truncate IMUSig to match the length of EMGSig
                IMUSig = IMUSig(:, 1:len_emg);
            end
            
            % Concatenate the signals vertically
            AllSig = [EMGSig; IMUSig];

            % Put signal and request processing
            this.signal.AppendSignal(size(AllSig, 2), AllSig);
        end
    end
end

