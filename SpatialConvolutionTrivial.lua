local SpatialConvolutionTrivial, parent = torch.class('ccn2.SpatialConvolutionTrivial', 'nn.Module')

function SpatialConvolutionTrivial:__init(nInputPlane, nOutputPlane)
   parent.__init(self)

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane

   self.weight = torch.Tensor(nInputPlane, nOutputPlane)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nInputPlane, nOutputPlane)
   self.gradBias = torch.Tensor(nOutputPlane)

   self.gradInput = torch.Tensor()
   self.output = torch.Tensor()

   self:reset()
--   self:cuda()
end

function SpatialConvolutionTrivial:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.nInputPlane)
   end
   self.weight:uniform(-stdv, stdv)
   self.bias:uniform(-stdv, stdv)   
end

function SpatialConvolutionTrivial:updateOutput(input)
   -- input is flattened (view)
   local tinput=input.new()
   local nMaps = input:size(1)
   local nPos = input:stride(1)
   local nOutMaps = self.nOutputPlane
   tinput:set(input:storage(), 1, torch.LongStorage{nMaps, nPos})
   
   -- MM
   self.output:resize(nOutMaps, nPos)
   self.ones=self.ones or input.new(nPos):fill(1)
   if self.ones:size(1) ~= nPos then self.ones= input.new(nPos):fill(1) end

   self.output:zero():addr(1, self.bias, self.ones)
   self.output:addmm(1, self.weight:t(), tinput)

   -- output is unflattened
   self.output:resize(nOutMaps, input:size(2), input:size(3), input:size(4))
   return self.output
end

function SpatialConvolutionTrivial:updateGradInput(input, gradOutput)
   -- gradOutput is flattened (view)
   local nMaps = input:size(1)
   local nPos = input:stride(1)
   local nOutMaps = self.nOutputPlane
   local tgradOutput=gradOutput.new()
   tgradOutput:set(gradOutput:storage(), 1, torch.LongStorage{nOutMaps, nPos})
  
   local nElement = self.gradInput:nElement()
   self.gradInput:resizeAs(input)

   self.gradInput:resizeAs(input)
   if self.gradInput:nElement() ~= nElement then
      self.gradInput:zero()
   end

   -- gradInput is flattened (view)
   local tgradInput=self.gradInput.new()
   tgradInput:set(self.gradInput:storage(), 1, torch.LongStorage{nMaps, nPos})

   tgradInput:addmm(0, 1, self.weight, tgradOutput)
   return self.gradInput
end

function SpatialConvolutionTrivial:accGradParameters(input, gradOutput, scale)
   -- input is flattened (view)
   local tinput=input.new()
   local nMaps = input:size(1)
   local nPos = input:stride(1)
   local nOutMaps = self.nOutputPlane
   tinput:set(input:storage(), 1, torch.LongStorage{nMaps, nPos})

   -- gradOutput is flattened (view)
   local nMaps = input:size(1)
   local nPos = input:stride(1)
   local nOutMaps = self.nOutputPlane
   local tgradOutput=gradOutput.new()
   tgradOutput:set(gradOutput:storage(), 1, torch.LongStorage{nOutMaps, nPos})
   
   self.gradWeight:addmm(scale, tinput, tgradOutput:t())
   self.gradBias:addmv(scale, tgradOutput, self.ones)
end


