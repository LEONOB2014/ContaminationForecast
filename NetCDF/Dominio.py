from netCDF4 import Dataset
import netCDF4 as nc4
import NewBBOX as ne

#net = Dataset('NewFile.nc');
net = Dataset('/home/pablo/DATA/Dom1_2017-06-13.nc')

variables=['Uat10','Vat10',PREC2];

LON = net.variables['Longitude'][:];
LAT = net.variables['Latitude'][:];
TIME = net.variables['Time'][:];

LONsize = len(LON);
LATsize = len(LAT);
TIMEsize = len(TIME);

minLON = min(LON);
maxLON = max(LON);
minLAT = min(LAT);
maxLAT = max(LAT);

print('El siguiente es el intervalo de latitudes disponible: \n');
print(minLAT);
print(maxLAT);

print('El siguiente es el intervalo de longitudes disponible \n');
print(minLON);
print(maxLON);
print('\n')


minlat=19.4284700;
maxlat=20;
minlon=-99.1276600;
maxlon=-97;

celda = [];
var_cut=[];
for i in variables:
    var= net.variables[i][:]
    celda.append(var);
    result = ne.NewBBOX(var,LON,LAT,LONsize,LATsize,minlat,maxlat,minlon,maxlon);
    var_cut.append(result[0]);

newLAT = result[1];
newLON = result[2];
NewFileName = 'NewFileVars.nc';

files = nc4.Dataset(NewFileName,'w',format='NETCDF4');

files.createDimension('newLON', len(newLON));
files.createDimension('newLAT',len(newLAT));
files.createDimension('TIME', TIMEsize);

varlon= files.createVariable('Longitude','float32','newLON');
varlat= files.createVariable('Latitude','float32','newLAT');
vartime= files.createVariable('Time','float32','TIME');

new_vars=[]
for i in range(len(variables)):
    var_temp = files.createVariable(variables[i],'float32', ('Time','newLON','newLAT'));
    new_vars.append(var_temp);

vartime[:] = TIME
varlon[:] = newLON;
varlat[:] = newLAT;

i = 0;
for x in range(len(new_vars)):
    new_vars[x] = var_cut[x];
    i+=1;

files.close();

f=nc4.Dataset('NewFileVars.nc','r');
print(f);
