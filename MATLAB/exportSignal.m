load("SavedData\gprdata.mat");
csvwrite("SavedCSV\signal.csv", signalCopy.signal);
csvwrite("SavedCSV\label.csv", signalCopy.tags);