std_msgs/Header header

string text
string template_id                      # grounding skeleton used
string source                           # "template" | "llm"
uint8  severity

string[] referenced_ids
builtin_interfaces/Time trigger_time

uint8 SEVERITY_INFO=0
uint8 SEVERITY_NOTICE=1
uint8 SEVERITY_ALERT=2
