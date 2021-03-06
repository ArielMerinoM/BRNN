require 'rnn'
require 'math'
require 'funciones'
require 'brnn'

--DECLARACION DE VARIABLES


rho = 10
lr = 0.00001
epocas = 100


--esta es la parte que hay que generalizar ***
carpetas = {'../Datos procesados/semana/','../Datos procesados/viernes/'}
archivos = {'03-MARZO','04-ABRIL','05-MAYO','06-JUNIO','07-JULIO','08-AGOSTO','09-SEPTIEMBRE','10-OCTUBRE','11-NOVIEMBRE','12-DICIEMBRE'}
FERIADOS = {{1,1},{2,2},{4,3},{4,21},{5,1},{9,7},{10,12},{11,2},{12,25}}

viajes = {}

for carpeta = 1, 2 do
	for archivo = 1, #archivos do
		viaje = {}
		entrada = io.open( carpetas[carpeta]..archivos[archivo]..'-I.txt','r')
		print(carpetas[carpeta]..archivos[archivo]..'-I.txt')
		linea = entrada:read()
		while linea ~= nil do
			linea = split(linea,'\t')
			if linea[1] == 'F' then
				T = 3600 * tonumber(linea[5]) + 60 * tonumber(linea[6]) + tonumber(linea[7])
				D = tonumber(linea[9])
				info = {tonumber(linea[2]),tonumber(linea[3]),tonumber(linea[5]),tonumber(linea[6]),tonumber(linea[7])} --dia, mes, hora, minuto, segundo
				table.insert(viaje,{T,D})
				table.insert(viaje, info)
				if #viaje <= 93 then --filtro de los 90 minutos
					if not feriado(FERIADOS, info[1], info[2]) then --filtro feriados
						table.insert(viajes, viaje)
					end
				end
				viaje = {}
			else
				dd = tonumber(linea[1])
				d = tonumber(linea[3])
				t = tonumber(linea[4])
				table.insert(viaje, {dd, d, t})
			end
			linea = entrada:read()
		end
		entrada:close()
	end
end

print(#viajes)

viajesX = {}
viajesY = {}
datosV = {}
for i = 1, #viajes do
	x = {}
	y = {}
	D = viajes[i][#(viajes[i])-1][2]
	T = viajes[i][#(viajes[i])-1][1]
	for j = 1, #viajes[i] - 3 do
		dd = viajes[i][j][1]/D
		d = viajes[i][j][2]/D
		t = (viajes[i][j][3] + T)/86400
		t0 = T/86400
		table.insert(x, {t0, t, d, dd})
		dtm1 = viajes[i][j+1][1]/D
		table.insert(y, dtm1*100) --el objetivo es adivinar el porcentaje de avance en el tiempo t+1
		--los resultados son amplificados x100 para obtener una medida porcentual
	end
	table.insert(viajesX, x)
	table.insert(viajesY, y)
	table.insert(datosV, viajes[i][#viajes[i]])
end

viajes = nil
collectgarbage()

--generacion de los datos de validacion


archivo_validacion = io.open('./conjunto_validacion.txt','r')
validacionX = {}
validacionY = {}

for i = 1, 3002 do
	linea = archivo_validacion:read()
	linea = split(linea,'\t')
	linea[1] = tonumber(linea[1])
	linea[2] = tonumber(linea[2])
	linea[3] = tonumber(linea[3])
	linea[4] = tonumber(linea[4])
	linea[5] = tonumber(linea[5])
	for j = 1, #datosV do
		if linea[1] == datosV[j][1] and linea[2] == datosV[j][2] and linea[3] == datosV[j][3] and linea[4] == datosV[j][4] and linea[5] == datosV[j][5] then
			table.insert(validacionX, viajesX[j])
			table.insert(validacionY, viajesY[j])
			table.remove(viajesX, j)
			table.remove(viajesY, j)
			table.remove(datosV, j)
			break
		end
	end
end

datosV = nil
collectgarbage()

--creacion de los folds
foldsX = {{},{},{},{},{},{},{},{},{},{}}
foldsY = {{},{},{},{},{},{},{},{},{},{}}

while #viajesX > 0 do
	for i = 1, 10 do
		if #viajesX > 0 then
			indice = math.random(#viajesX)
			table.insert(foldsX[i], viajesX[indice])
			table.insert(foldsY[i], viajesY[indice])
			table.remove(viajesX, indice)
			table.remove(viajesY, indice)
		else
			break
		end

	end
end

viajesX = nil
viajesY = nil
collectgarbage()

for i = 1, #foldsX do
	print(#foldsX[i])
end


--hasta este punto, foldsX y foldsY contienen los datos de entrenamiento
--y en validacionX y validacionY estan almacenados los datos de prueba

--generacion de la red
fwd = nn.LSTM(4,100,rho)
bwd = nn.LSTM(4,100,rho)
brnn = nn.BRNN(fwd,bwd)
red = nn.Sequential()
red:add(brnn)
red:add(nn.Linear(100,100))
red:add(nn.Linear(100,100))
red:add(nn.Linear(100,1))


--criterio de aprendizaje = MSE
criterion = nn.MSECriterion()

--hasta este punto esta bien, la red esta constituida y los datos son correctos

--k folds

--prueba inicial

for i = 1, #validacionX do
	validacionX[i] = torch.DoubleTensor(validacionX[i]) 
	validacionY[i] = torch.DoubleTensor(validacionY[i])
end

tiempo0 = os.time()

errPromedio = 0
for i = 1, #validacionY do
	resultado = red:forward(validacionX[i])
	errPromedio = errPromedio + criterion:forward(resultado, validacionY[i])
	red:forget()
end

errPromedio =  errPromedio / #validacionY

registro = io.open('./registro/log.txt','a')
registro:write('epoca\tfold\terror[prom MSE]\ttiempo[s]\n')
registro:write('0\t0\t'..errPromedio..'\t'..os.time() - tiempo0..'\n')
registro:close()



for epoca = 1, epocas do
	for i = 1, 10 do
		--entrenamiento con el fold i
		for j = 1, #(foldsX[i]) do --todos los casos en un fold
			os.execute('clear')
			print('epoca:',epoca)
			print('fold: '..i..' - avance: '..math.floor(100*j/#(foldsY[i])).."%")
			print('tiempo transcurrido: '..os.time() - tiempo0..'s')
			print('MSE promedio: '..errPromedio)
			entradas = foldsX[i][j] --todos los puntos de un recorrido
			objetivos = foldsY[i][j]

			serieX = {}
			serieY = {}

			for k = 1, #entradas do
				table.insert(serieX, entradas[k])
				table.insert(serieY, objetivos[k])

				if #serieX >= 10 then
					gradientUpgrade(red, torch.DoubleTensor(serieX), torch.DoubleTensor(serieY), criterion, lr)
					red:forget()
				end
			end
		end
		os.execute('clear')
		print('epoca:',epoca)
		print('fold: '..i..' - avance: '.."100%")
		print('tiempo transcurrido: '..os.time() - tiempo0..'s')
		print('Guardando...')
		torch.save('./redes/bidirectionalLSTM.rn', red)
		print('Listo!!!')

		--pruebas

		errPromedio = 0
		for j = 1, #validacionX do
			resultado = red:forward(validacionX[j])
			errPromedio = errPromedio + criterion:forward(resultado, validacionY[j])
			red:forget()
		end
		errPromedio = errPromedio / #validacionY

		--hay que guardar un registro con errPromedio

		registro = io.open('./registro/log.txt','a')
		registro:write(epoca..'\t'..i..'\t'..errPromedio..'\t'..os.time() - tiempo0..'\n')
		registro:close()
	end
	shuffle(foldsX, foldsY)
end

os.execute('clear')
print('avance general: 100%')
print('entrenamiento terminado')
