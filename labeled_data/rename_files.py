import os
from glob import glob

shots = [97927, 97803, 96532, 97828, 97830, 97832, 97835, 97971, 97465, 97466, 97468, 97469, 97473, 94785, 97476, 97477, 96713, 97745, 98005, 96993, 96745, 94315, 97396, 96885, 97398, 97399, 94968, 94969, 94971, 94973]

for x in shots:
    #a = glob("../data/Validated/TCV_" + str(x) + "_marceca_labeled.csv")
    #command = "mv " + a[0] + " /Lac8_D/DISTOOL/JET/Event_Detection/TCV_Validation/sensitive_validation/LHD/157_shots/" + "TCV_" + str(x) + "_apau_and_marceca_labeled.csv"
    if os.path.exists("./JET/detrend_5kHz/" + "JET_" + str(x) + "_detrend_5kHz_labeled.csv"):
        command = "mv " + " ./JET/detrend_5kHz/" + "JET_" + str(x) + "_detrend_5kHz_labeled.csv ./JET/detrend_5kHz/" + "JET_" + str(x) + "_detrend_labeled.csv"
        print(command)
        os.system(command)
