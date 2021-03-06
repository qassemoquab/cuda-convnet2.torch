local C = ccn2.C

local SpatialConvolution, parent = torch.class('ccn2.SpatialConvolution', 'nn.Module')

function SpatialConvolution:__init(nInputPlane, nOutputPlane, kH, dH, padding, groups)
   parent.__init(self)

   dH = dH or 1 -- stride
   padding = padding or 0
   groups = groups or 1

   if not (nInputPlane >= 1 and (nInputPlane <= 3 or math.fmod(nInputPlane, 4) == 0)) then
      error('Assertion failed: [(nInputPlane >= 1 and (nInputPlane <= 3 or math.fmod(nInputPlane, 4)))]. Number of input channels has to be 1, 2, 3 or a multiple of 4')
   end
   if math.fmod(nOutputPlane, 16) ~= 0 then
      error('Assertion failed: [math.fmod(nOutputPlane, 16) == 0]. Number of output planes has to be a multiple of 16.')
   end

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kH = kH
   self.dH = dH
   self.groups = groups
   self.padding = padding

   self.weight = torch.Tensor(nInputPlane*kH*kH/groups, nOutputPlane)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nInputPlane*kH*kH/groups, nOutputPlane)
   self.gradBias = torch.Tensor(nOutputPlane)

   self.gradInput = torch.Tensor()
   self.output = torch.Tensor()

   self:reset()
--   self:cuda()
end

function SpatialConvolution:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kH*self.kH*self.nInputPlane/self.groups)
   end
   self.weight:uniform(-stdv, stdv)
   self.bias:uniform(-stdv, stdv)   
end

function SpatialConvolution:updateOutput(input)
   ccn2.typecheck(input)
   ccn2.inputcheck(input)
   local nBatch = input:size(4)
   local oH = math.ceil((self.padding * 2 + input:size(2) - self.kH) / self.dH + 1);
   local inputC = input:view(input:size(1) * input:size(2) * input:size(3), 
                             input:size(4))
   self.groups = self.groups or 1

   -- do convolution
   C['convFilterActs'](inputC:cdata(), self.weight:cdata(), self.output:cdata(), 
                       input:size(2), oH, oH, 
                          -self.padding, self.dH, self.nInputPlane, self.groups);
   -- add bias
   self.output = self.output:view(self.nOutputPlane, oH*oH*nBatch)
   C['addBias'](self.output:cdata(), self.bias:cdata());
   self.output = self.output:view(self.nOutputPlane, oH, oH, nBatch)
   return self.output
end

function SpatialConvolution:updateGradInput(input, gradOutput)
   ccn2.typecheck(input); ccn2.typecheck(gradOutput); 
   ccn2.inputcheck(input); ccn2.inputcheck(gradOutput);
   local oH = gradOutput:size(2); 
   local iH = input:size(2)
   local nBatch = input:size(4)
   self.gradInput:resize(self.nInputPlane*iH*iH, nBatch);
   local gradOutputC = gradOutput:view(
      gradOutput:size(1) * gradOutput:size(2) * gradOutput:size(3), gradOutput:size(4)
   )
   C['convImgActs'](gradOutputC:cdata(), self.weight:cdata(), self.gradInput:cdata(), 
                    iH, iH, oH, 
                       -self.padding, self.dH, self.nInputPlane, self.groups);
   self.gradInput = self.gradInput:view(self.nInputPlane, iH, iH, nBatch)
   return self.gradInput
end

function SpatialConvolution:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   ccn2.typecheck(input); ccn2.typecheck(gradOutput); 
   ccn2.inputcheck(input); ccn2.inputcheck(gradOutput);
   local oH = gradOutput:size(2);
   local iH = input:size(2)
   local nBatch = input:size(4)
   local inputC = input:view(input:size(1) * input:size(2) * input:size(3), input:size(4))
   local gradOutputC = gradOutput:view(gradOutput:size(1) * gradOutput:size(2) * gradOutput:size(3), gradOutput:size(4))
   local sumWidth = oH
   C['convWeightActsSt'](inputC:cdata(), gradOutputC:cdata(), self.gradWeight:cdata(),
                         iH, oH, oH, self.kH, 
                            -self.padding, self.dH, self.nInputPlane, self.groups, sumWidth, 0, scale);
   gradOutputC = gradOutput:view(self.nOutputPlane, oH * oH * nBatch)
   C['gradBias'](gradOutputC:cdata(), self.gradBias:cdata(), scale);   
end


