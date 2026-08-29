std_msgs/Header header

string    formation_class               # one of the 8 classes
float64   confidence
float64[] class_probabilities           # length 8, ordered per FORMATION_CLASSES
string[]  member_ids                    # global_ids participating
bool      from_truth                    # true when fed parametric truth (debug)
