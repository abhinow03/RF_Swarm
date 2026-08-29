std_msgs/Header header

string global_id

geometry_msgs/Point[]     waypoints
builtin_interfaces/Time[] waypoint_times    # same length as waypoints
float64[]                 position_sigma    # metres, 1-sigma radius per waypoint

float64 horizon_sec
string  model_id                        # "cv_baseline" | "<ml model tag>"
