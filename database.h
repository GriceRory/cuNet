struct database{
  vector **inputs;
  vector **outputs;
  int size;
};


//memory
void buildDatabase(database *db, file f);
void saveDatabase(database *db, file f);
void copyHostToDevice(database *host, database *device);
void copyDeviceToHost(database *device, database *host);
