
classdef ROS2PublisherWithClass < handle
    properties
        node
        publisher
        data_publisher
        publishTimer
        i
    end
    
    methods
        function this = ROS2PublisherWithClass()
            % Create a ROS 2 node
            this.node = ros2node("/test_node");
            % Create a publisher on the specified topic
            this.publisher = ros2publisher(this.node, '/float64emg', 'std_msgs/Float64MultiArray');
            this.data_publisher = ros2publisher(this.node, '/float64imu', 'std_msgs/Float64MultiArray');
            %this.msg = ros2message(this.publisher);
            %this.msg.data = [5.555 6.666 7.777 8.888; 1.111 2.222 3.333 4.444;5.555 6.666 7.777 8.888; 1.111 2.222 3.333 4.444;5.555 6.666 7.777 8.888; 1.111 2.222 3.333 4.444;5.555 6.666 7.777 8.888; 1.111 2.222 3.333 4.444;5.555 6.666 7.777 8.888; 1.111 2.222 3.333 4.444;5.555 6.666 7.777 8.888; 1.111 2.222 3.333 4.444;5.555 6.666 7.777 8.888; 1.111 2.222 3.333 4.444;5.555 6.666 7.777 8.888; 1.111 2.222 3.333 4.444;5.555 6.666 7.777 8.888; 1.111 2.222 3.333 4.444;5.555 6.666 7.777 8.888; 1.111 2.222 3.333 4.444;5.555 6.666 7.777 8.888; 1.111 2.222 3.333 4.444;5.555 6.666 7.777 8.888; 1.111 2.222 3.333 4.444;5.555 6.666 7.777 8.888; 1.111 2.222 3.333 4.444;5.555 6.666 7.777 8.888; 1.111 2.222 3.333 4.444;5.555 6.666 7.777 8.888]';
            this.i = 0;
            % Create a timer to publish message every second
            this.publishTimer = timer('ExecutionMode', 'fixedRate', 'Period', 0.05, ...
                'TimerFcn', @this.publishMessage);
            % Start the timer
            start(this.publishTimer);
            % Pause for 10 seconds
            pause(900);
            % Call the cleanup function
            this.cleanup();
        end
        
        function publishMessage(this,~,~)
            % Create a ROS 2 message
            msg1 = ros2message(this.publisher);
            msg2 = ros2message(this.data_publisher);
            msg1.data = 100 * rand(1, 1800);
            msg2.data = 100 * rand(1, 3600);
            %msg1.data = [5.555 6.666 7.777 8.888; 1.111 2.222 3.333 4.444;5.555 6.666 7.777 8.888; 1.111 2.222 3.333 4.444;5.555 6.666 7.777 8.888; 1.111 2.222 3.333 4.444;5.555 6.666 7.777 8.888; 1.111 2.222 3.333 4.444;5.555 6.666 7.777 8.888; 1.111 2.222 3.333 4.444;5.555 6.666 7.777 8.888]';
            %msg2.data = [5.555 6.666 7.777 8.888; 1.111 2.222 3.333 4.444;5.555 6.666 7.777 8.888; 1.111 2.222 3.333 4.444;5.555 6.666 7.777 8.888; 1.111 2.222 3.333 4.444;5.555 6.666 7.777 8.888; 1.111 2.222 3.333 4.444;5.555 6.666 7.777 8.888; 1.111 2.222 3.333 4.444;5.555 6.666 7.777 8.888; 1.111 2.222 3.333 4.444;5.555 6.666 7.777 8.888; 1.111 2.222 3.333 4.444;5.555 6.666 7.777 8.888; 1.111 2.222 3.333 4.444;5.555 6.666 7.777 8.888; 1.111 2.222 3.333 4.444]';
            % Publish the message: it starts from the first column (all the
            % rows) and then pass to the second column..
            send(this.publisher, msg1);
            send(this.data_publisher, msg2);
            this.i = this.i + 1;
            %disp(msg.data);
        end
        
        function cleanup(this)
            % Stop and delete the timer
            stop(this.publishTimer);
            delete(this.publishTimer);
            stop(this.data_publisher);
            delete(this.data_publisher);
            % Shutdown the ROS 2 node
            clear this.node;
        end
    end
end
