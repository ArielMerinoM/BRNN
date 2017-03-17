require 'rnn'
require 'math'
------------------------------------------------------------------------
--[[ ReverseSequence ]] --
-- Reverses a sequence on a given dimension.
-- Example: Given a tensor of torch.Tensor({{1,2,3,4,5}, {6,7,8,9,10})
-- nn.ReverseSequence(1):forward(tensor) would give: torch.Tensor({{6,7,8,9,10},{1,2,3,4,5}})
------------------------------------------------------------------------
local ReverseSequence, parent = torch.class("nn.ReverseSequence", "nn.Module")

function ReverseSequence:__init(dim)
    parent.__init(self)
    self.output = torch.Tensor()
    self.gradInput = torch.Tensor()
    self.outputIndices = torch.LongTensor()
    self.gradIndices = torch.LongTensor()
    assert(dim, "Must specify dimension to reverse sequence over")
    assert(dim <= 3, "Dimension has to be no greater than 3 (Only supports up to a 3D Tensor).")
    self.dim = dim
end

function ReverseSequence:reverseOutput(input)
    self.output:resizeAs(input)
    self.outputIndices:resize(input:size())
    local T = input:size(1)
    for x = 1, T do
        self.outputIndices:narrow(1, x, 1):fill(T - x + 1)
    end
    self.output:gather(input, 1, self.outputIndices)
end

function ReverseSequence:updateOutput(input)
    if (self.dim == 1) then
        self:reverseOutput(input)
    end
    if (self.dim == 2) then
        input = input:transpose(1, 2)
        self:reverseOutput(input)
        self.output = self.output:transpose(1, 2)
    end
    if (self.dim == 3) then
        input = input:transpose(1, 3)
        self:reverseOutput(input)
        self.output = self.output:transpose(1, 3)
    end
    return self.output
end

function ReverseSequence:reverseGradOutput(gradOutput)
    self.gradInput:resizeAs(gradOutput)
    self.gradIndices:resize(gradOutput:size())
    local T = gradOutput:size(1)
    for x = 1, T do
        self.gradIndices:narrow(1, x, 1):fill(T - x + 1)
    end
    self.gradInput:gather(gradOutput, 1, self.gradIndices)
end

function ReverseSequence:updateGradInput(inputTable, gradOutput)
    if (self.dim == 1) then
        self:reverseGradOutput(gradOutput)
    end
    if (self.dim == 2) then
        gradOutput = gradOutput:transpose(1, 2)
        self:reverseGradOutput(gradOutput)
        self.gradInput = self.gradInput:transpose(1, 2)
    end
    if (self.dim == 3) then
        gradOutput = gradOutput:transpose(1, 3)
        self:reverseGradOutput(gradOutput)
        self.gradInput = self.gradInput:transpose(1, 3)
    end
    return self.gradInput
end

------------------------------------------------------------------------
--[[ BRNN ]] --
-- Encapsulates a forward, backward and merge module.
-- Input is a tensor e.g batch x time x inputdim.
-- Output is a tensor of the same length e.g batch x time x outputdim.
-- Applies a forward rnn to input tensor in forward order
-- and applies a backward rnn in reverse order.
-- Reversal of the sequence happens on the chosen dim (defaults to 2).
-- For each step, the outputs of both rnn are merged together using
-- the merge module (defaults to nn.CAddTable() which sums the activations).
------------------------------------------------------------------------

local BRNN, parent = torch.class('nn.BRNN', 'nn.Container')

function BRNN:__init(forward, backward, merge, dimToReverse)
    if not torch.isTypeOf(forward, 'nn.Module') then
        error "BRNN: expecting nn.Module instance at arg 1"
    end
    self.forwardModule = forward
    self.backwardModule = backward
    self.merge = merge
    self.dim = dimToReverse
    if not self.backwardModule then
        self.backwardModule = forward:clone()
        self.backwardModule:reset()
    end
    if not torch.isTypeOf(self.backwardModule, 'nn.Module') then
        error "BRNN: expecting nn.Module instance at arg 2"
    end
    if not self.merge then
        self.merge = nn.CAddTable()
    end
    if not self.dim then
        self.dim = 1 -- default to second dimension to reverse (expecting batch x time x inputdim).
        --se necesita que la reversa reordene la entrada segun la primera dimension para que la brnn funcione segun el tiempo
    end
    local backward = nn.Sequential()
    backward:add(nn.ReverseSequence(self.dim)) -- reverse
    backward:add(self.backwardModule)
    backward:add(nn.ReverseSequence(self.dim)) -- unreverse

    local concat = nn.ConcatTable()
    concat:add(self.forwardModule):add(backward)

    local brnn = nn.Sequential()
    brnn:add(concat)
    brnn:add(self.merge)

    parent.__init(self)

    self.output = torch.Tensor()
    self.gradInput = torch.Tensor()

    self.module = brnn
    -- so that it can be handled like a Container
    self.modules[1] = brnn
end

function BRNN:updateOutput(input)
    self.output = self.module:updateOutput(input)
    return self.output
end

function BRNN:updateGradInput(input, gradOutput)
    self.gradInput = self.module:updateGradInput(input, gradOutput)
    return self.gradInput
end

function BRNN:accGradParameters(input, gradOutput, scale)
    self.module:accGradParameters(input, gradOutput, scale)
end

function BRNN:accUpdateGradParameters(input, gradOutput, lr)
    self.module:accUpdateGradParameters(input, gradOutput, lr)
end

function BRNN:sharedAccUpdateGradParameters(input, gradOutput, lr)
    self.module:sharedAccUpdateGradParameters(input, gradOutput, lr)
end

function BRNN:__tostring__()
    if self.module.__tostring__ then
        return torch.type(self) .. ' @ ' .. self.module:__tostring__()
    else
        return torch.type(self) .. ' @ ' .. torch.type(self.module)
    end
end