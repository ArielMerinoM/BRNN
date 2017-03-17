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

for i = 1, #viajes do
	x = {}
	y = {}
	D = viajes[i][#(viajes[i])-1][2]
	T = viajes[i][#(viajes[i])-1][1]
	print(D,T)
	for j = 1, #viajes[i] - 3 do
		dd = viajes[i][j][1]/D
		d = viajes[i][j][2]/D
		t = (viajes[i][j][3] + T)/86400
		t0 = T/86400
		table.insert(x, {t0, t, d, dd})
		dtm1 = viajes[i][j+1][2]/D
		table.insert(y, dtm1*100) --el objetivo es adivinar el porcentaje de avance en el tiempo t+1
		--los resultados son amplificados x100 para obtener una medida porcentual
	end
	table.insert(viajesX, x)
	table.insert(viajesY, y)
end

viajes = nil
collectgarbage()

--[[
--hasta aqui vamos super bien !!!

--creacion de los folds
foldsX = {}
foldsY = {}

for i = 1, 11 do
	table.insert(foldsX, {})
	table.insert(foldsY, {})
	for j = 1, 1396 do
		eleccion = choice(viajesX, viajesY)
		table.insert(foldsX[i], eleccion[1])
		table.insert(foldsY[i], eleccion[2])
		eleccion = {}
	end
end

validacionX = table.remove(foldsX, 11)
validacionY = table.remove(foldsY, 11)


--hasta este punto, foldsX y foldsY contienen los datos de entrenamiento
--y en validacionX y validacionY estan almacenados los datos de prueba

--generacion de la red
fwd = nn.LSTM(4,100,rho)
bwd = nn.LSTM(4,100,rho)
brnn = nn.BRNN(fwd,bwd)
red = nn.Sequential()
red:add(brnn)
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
			print('avance general: '.. math.floor(100 * (epoca - 1) / epocas) .. "%")
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
		print('avance general: '.. math.floor(100 * (epoca - 1) / epocas) .. "%")
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

]]