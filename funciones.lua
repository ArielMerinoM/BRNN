require 'math'
require 'rnn'
function gradientUpgrade(model, x, y, criterion, learningRate )
	local prediction = model:forward(x)
	local err = criterion:forward(prediction, y)
	local gradOutputs = criterion:backward(prediction, y)
	model:backward(x, gradOutputs)
	model:updateParameters(learningRate)
	model:zeroGradParameters()

	return err
end

function split(inputstr, sep)
    if sep == nil then
            sep = "%s"
    end
    local t={} ; i=1
    for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
            t[i] = str
            i = i + 1
    end
    return t
end

function choice(tabla1, tabla2)
	if #tabla1 == 0 then
		return nil
	end
	indice = math.random(#tabla1)
	return {table.remove(tabla1, indice), table.remove(tabla2, indice)}
end

function shuffle(tabla1, tabla2)
	cantidad = #tabla1
	for i = 1, cantidad do
		indice = math.random(cantidad)
		elemento1 = table.remove(tabla1, indice)
		elemento2 = table.remove(tabla2, indice)
		table.insert(tabla1, elemento1)
		table.insert(tabla2, elemento2)
	end
end