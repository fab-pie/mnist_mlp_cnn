
const mnist_convnet = (() => {
const getTensorBuffer = (safetensorBuffer, tensorMetadata) => {
  return safetensorBuffer.subarray(...tensorMetadata.data_offsets);
};

const getTensorMetadata = (safetensorBuffer) => {
    const metadataLength = Number(new DataView(safetensorBuffer.buffer).getBigUint64(0, true));
    const metadata = JSON.parse(new TextDecoder("utf8").decode(safetensorBuffer.subarray(8, 8 + metadataLength)));
    return Object.fromEntries(Object.entries(metadata).filter(([k, v]) => k !== "__metadata__").map(([k, v]) => [k, {...v, data_offsets: v.data_offsets.map(x => 8 + metadataLength + x)}]));
};

const createEmptyBuf = (device, size) => {
    return device.createBuffer({size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
};

const createUniformBuf = (device, size) => {
  return device.createBuffer({size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST})
}

const createInfinityUniformBuf = (device) => {
  const size = 4;
  const buf = device.createBuffer({
    mappedAtCreation: true,
    size,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  });
  new Float32Array(buf.getMappedRange())[0] = Infinity;
  buf.unmap();
  return buf;
};

const createWeightBuf = (device, size, data) => {
  const buf = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
  new Uint8Array(buf.getMappedRange()).set(data); buf.unmap();
  return buf;
};

const addComputePass = (device, commandEncoder, pipeline, layout, infinityUniformBuf, bufs, workgroup) => {
  const bindGroup = device.createBindGroup({
    layout: layout,
    entries: [
      { binding: 0, resource: { buffer: infinityUniformBuf } },
      ...bufs.map((buffer, index) => ({ binding: index + 1, resource: { buffer } }))
    ]
  });

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(...workgroup);
  passEncoder.end();
};

const r_4_3_2_8_8_3_4_5_5 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_18432:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_784:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_800:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_32:array<f32>;
@compute @workgroup_size(2,8,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,12>;
  var gidx0 = i32(gindex.x); /* 3 */
  var gidx1 = i32(gindex.y); /* 4 */
  var lidx0 = i32(lindex.x); /* 2 */
  var lidx1 = i32(lindex.y); /* 8 */
  var lidx2 = i32(lindex.z); /* 8 */
  var alu0 = (lidx2*3);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  for (var ridx1007 = 0; ridx1007 < 5; ridx1007++) {
    var alu13 = ((gidx1*200)+(lidx0*100)+(ridx1007*5));
    var val0 = data2_800[alu13];
    var val1 = data2_800[(alu13+1)];
    var val2 = data2_800[(alu13+2)];
    var val3 = data2_800[(alu13+3)];
    var val4 = data2_800[(alu13+4)];
    var val5 = data2_800[(alu13+25)];
    var val6 = data2_800[(alu13+26)];
    var val7 = data2_800[(alu13+27)];
    var val8 = data2_800[(alu13+28)];
    var val9 = data2_800[(alu13+29)];
    var val10 = data2_800[(alu13+50)];
    var val11 = data2_800[(alu13+51)];
    var val12 = data2_800[(alu13+52)];
    var val13 = data2_800[(alu13+53)];
    var val14 = data2_800[(alu13+54)];
    var val15 = data2_800[(alu13+75)];
    var val16 = data2_800[(alu13+76)];
    var val17 = data2_800[(alu13+77)];
    var val18 = data2_800[(alu13+78)];
    var val19 = data2_800[(alu13+79)];
    var alu14 = ((gidx0*224)+(lidx1*28)+alu0+(ridx1007*28));
    var val20 = data1_784[alu14];
    var val21 = data1_784[(alu14+1)];
    var val22 = data1_784[(alu14+2)];
    var val23 = data1_784[(alu14+3)];
    var val24 = data1_784[(alu14+4)];
    var val25 = data1_784[(alu14+5)];
    var val26 = data1_784[(alu14+6)];
    acc0[4] = (acc0[4]+(val21*val0)+(val22*val1)+(val23*val2)+(val24*val3)+(val25*val4));
    acc0[5] = (acc0[5]+(val21*val5)+(val22*val6)+(val23*val7)+(val24*val8)+(val25*val9));
    acc0[6] = (acc0[6]+(val21*val10)+(val22*val11)+(val23*val12)+(val24*val13)+(val25*val14));
    acc0[7] = (acc0[7]+(val21*val15)+(val22*val16)+(val23*val17)+(val24*val18)+(val25*val19));
    acc0[8] = (acc0[8]+(val22*val0)+(val23*val1)+(val24*val2)+(val25*val3)+(val26*val4));
    acc0[9] = (acc0[9]+(val22*val5)+(val23*val6)+(val24*val7)+(val25*val8)+(val26*val9));
    acc0[10] = (acc0[10]+(val22*val10)+(val23*val11)+(val24*val12)+(val25*val13)+(val26*val14));
    acc0[11] = (acc0[11]+(val22*val15)+(val23*val16)+(val24*val17)+(val25*val18)+(val26*val19));
    acc0[1] = (acc0[1]+(val20*val5)+(val21*val6)+(val22*val7)+(val23*val8)+(val24*val9));
    acc0[2] = (acc0[2]+(val20*val10)+(val21*val11)+(val22*val12)+(val23*val13)+(val24*val14));
    acc0[3] = (acc0[3]+(val20*val15)+(val21*val16)+(val22*val17)+(val23*val18)+(val24*val19));
    acc0[0] = (acc0[0]+(val20*val0)+(val21*val1)+(val22*val2)+(val23*val3)+(val24*val4));
  }
  var precast0 = gidx1;
  var precast1 = lidx0;
  var precast2 = (bitcast<u32>(precast0)<<3u);
  var precast3 = (bitcast<u32>(precast1)<<2u);
  var alu28 = (bitcast<i32>(precast2)+bitcast<i32>(precast3));
  var val27 = data3_32[alu28];
  var val28 = data3_32[(alu28+1)];
  var val29 = data3_32[(alu28+2)];
  var val30 = data3_32[(alu28+3)];
  var alu29 = (acc0[0]+val27);
  var alu30 = (acc0[1]+val28);
  var alu31 = (acc0[2]+val29);
  var alu32 = (acc0[3]+val30);
  var alu33 = (acc0[4]+val27);
  var alu34 = (acc0[5]+val28);
  var alu35 = (acc0[6]+val29);
  var alu36 = (acc0[7]+val30);
  var alu37 = (acc0[8]+val27);
  var alu38 = (acc0[9]+val28);
  var alu39 = (acc0[10]+val29);
  var alu40 = (acc0[11]+val30);
  var alu41 = ((gidx0*192)+(gidx1*4608)+(lidx0*2304)+(lidx1*24)+alu0);
  data0_18432[alu41] = (alu29*(1/(1.0f+exp2((alu29*-1.4426950408889634f)))));
  data0_18432[(alu41+576)] = (alu30*(1/(1.0f+exp2((alu30*-1.4426950408889634f)))));
  data0_18432[(alu41+1152)] = (alu31*(1/(1.0f+exp2((alu31*-1.4426950408889634f)))));
  data0_18432[(alu41+1728)] = (alu32*(1/(1.0f+exp2((alu32*-1.4426950408889634f)))));
  data0_18432[(alu41+1)] = (alu33*(1/(1.0f+exp2((alu33*-1.4426950408889634f)))));
  data0_18432[(alu41+577)] = (alu34*(1/(1.0f+exp2((alu34*-1.4426950408889634f)))));
  data0_18432[(alu41+1153)] = (alu35*(1/(1.0f+exp2((alu35*-1.4426950408889634f)))));
  data0_18432[(alu41+1729)] = (alu36*(1/(1.0f+exp2((alu36*-1.4426950408889634f)))));
  data0_18432[(alu41+2)] = (alu37*(1/(1.0f+exp2((alu37*-1.4426950408889634f)))));
  data0_18432[(alu41+578)] = (alu38*(1/(1.0f+exp2((alu38*-1.4426950408889634f)))));
  data0_18432[(alu41+1154)] = (alu39*(1/(1.0f+exp2((alu39*-1.4426950408889634f)))));
  data0_18432[(alu41+1730)] = (alu40*(1/(1.0f+exp2((alu40*-1.4426950408889634f)))));
}`;

const r_5_5_8_4_4_4_32_5_5 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_12800:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_18432:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_25600:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_32:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 5 */
  var gidx1 = i32(gindex.y); /* 5 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var precast0 = gidx0;
  var precast1 = (bitcast<u32>(precast0)<<2u);
  var cast0 = bitcast<i32>(precast1);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  for (var ridx1006 = 0; ridx1006 < 32; ridx1006++) {
    for (var ridx1007 = 0; ridx1007 < 5; ridx1007++) {
      var alu16 = (cast0+(gidx1*96)+(lidx1*24)+(ridx1006*576)+(ridx1007*24));
      var val0 = data1_18432[alu16];
      var val1 = data1_18432[(alu16+1)];
      var val2 = data1_18432[(alu16+2)];
      var val3 = data1_18432[(alu16+3)];
      var val4 = data1_18432[(alu16+4)];
      var val5 = data1_18432[(alu16+5)];
      var val6 = data1_18432[(alu16+6)];
      var val7 = data1_18432[(alu16+7)];
      var alu17 = ((lidx0*3200)+(ridx1006*25)+(ridx1007*5));
      var val8 = data2_25600[alu17];
      var val9 = data2_25600[(alu17+1)];
      var val10 = data2_25600[(alu17+2)];
      var val11 = data2_25600[(alu17+3)];
      var val12 = data2_25600[(alu17+4)];
      var val13 = data2_25600[(alu17+800)];
      var val14 = data2_25600[(alu17+801)];
      var val15 = data2_25600[(alu17+802)];
      var val16 = data2_25600[(alu17+803)];
      var val17 = data2_25600[(alu17+804)];
      var val18 = data2_25600[(alu17+1600)];
      var val19 = data2_25600[(alu17+1601)];
      var val20 = data2_25600[(alu17+1602)];
      var val21 = data2_25600[(alu17+1603)];
      var val22 = data2_25600[(alu17+1604)];
      var val23 = data2_25600[(alu17+2400)];
      var val24 = data2_25600[(alu17+2401)];
      var val25 = data2_25600[(alu17+2402)];
      var val26 = data2_25600[(alu17+2403)];
      var val27 = data2_25600[(alu17+2404)];
      acc0[0] = (acc0[0]+(val0*val8)+(val1*val9)+(val2*val10)+(val3*val11)+(val4*val12));
      acc0[1] = (acc0[1]+(val0*val13)+(val1*val14)+(val2*val15)+(val3*val16)+(val4*val17));
      acc0[2] = (acc0[2]+(val0*val18)+(val1*val19)+(val2*val20)+(val3*val21)+(val4*val22));
      acc0[3] = (acc0[3]+(val0*val23)+(val1*val24)+(val2*val25)+(val3*val26)+(val4*val27));
      acc0[4] = (acc0[4]+(val1*val8)+(val2*val9)+(val3*val10)+(val4*val11)+(val5*val12));
      acc0[5] = (acc0[5]+(val1*val13)+(val2*val14)+(val3*val15)+(val4*val16)+(val5*val17));
      acc0[6] = (acc0[6]+(val1*val18)+(val2*val19)+(val3*val20)+(val4*val21)+(val5*val22));
      acc0[7] = (acc0[7]+(val1*val23)+(val2*val24)+(val3*val25)+(val4*val26)+(val5*val27));
      acc0[8] = (acc0[8]+(val2*val8)+(val3*val9)+(val4*val10)+(val5*val11)+(val6*val12));
      acc0[9] = (acc0[9]+(val2*val13)+(val3*val14)+(val4*val15)+(val5*val16)+(val6*val17));
      acc0[10] = (acc0[10]+(val2*val18)+(val3*val19)+(val4*val20)+(val5*val21)+(val6*val22));
      acc0[11] = (acc0[11]+(val2*val23)+(val3*val24)+(val4*val25)+(val5*val26)+(val6*val27));
      acc0[12] = (acc0[12]+(val3*val8)+(val4*val9)+(val5*val10)+(val6*val11)+(val7*val12));
      acc0[13] = (acc0[13]+(val3*val13)+(val4*val14)+(val5*val15)+(val6*val16)+(val7*val17));
      acc0[14] = (acc0[14]+(val3*val18)+(val4*val19)+(val5*val20)+(val6*val21)+(val7*val22));
      acc0[15] = (acc0[15]+(val3*val23)+(val4*val24)+(val5*val25)+(val6*val26)+(val7*val27));
    }
  }
  var precast2 = lidx0;
  var precast3 = (bitcast<u32>(precast2)<<2u);
  var cast1 = bitcast<i32>(precast3);
  var val28 = data3_32[cast1];
  var val29 = data3_32[(cast1+1)];
  var val30 = data3_32[(cast1+2)];
  var val31 = data3_32[(cast1+3)];
  var alu36 = (cast0+(gidx1*80)+(lidx0*1600)+(lidx1*20));
  data0_12800[alu36] = (acc0[0]+val28);
  data0_12800[(alu36+1)] = (acc0[4]+val28);
  data0_12800[(alu36+2)] = (acc0[8]+val28);
  data0_12800[(alu36+3)] = (acc0[12]+val28);
  data0_12800[(alu36+400)] = (acc0[1]+val29);
  data0_12800[(alu36+401)] = (acc0[5]+val29);
  data0_12800[(alu36+402)] = (acc0[9]+val29);
  data0_12800[(alu36+403)] = (acc0[13]+val29);
  data0_12800[(alu36+800)] = (acc0[2]+val30);
  data0_12800[(alu36+801)] = (acc0[6]+val30);
  data0_12800[(alu36+802)] = (acc0[10]+val30);
  data0_12800[(alu36+803)] = (acc0[14]+val30);
  data0_12800[(alu36+1200)] = (acc0[3]+val31);
  data0_12800[(alu36+1201)] = (acc0[7]+val31);
  data0_12800[(alu36+1202)] = (acc0[11]+val31);
  data0_12800[(alu36+1203)] = (acc0[15]+val31);
}`;

const r_5_5_8_2_2_4_2_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_3200:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_12800:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_32:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_32:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_32:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_32:array<f32>;
@compute @workgroup_size(8,2,2) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 5 */
  var gidx1 = i32(gindex.y); /* 5 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 2 */
  var lidx2 = i32(lindex.z); /* 2 */
  var precast0 = gidx0;
  var precast1 = lidx0;
  var precast2 = lidx2;
  var cast0 = bitcast<u32>(precast0);
  var precast3 = (cast0<<2u);
  var precast4 = (bitcast<u32>(precast1)<<2u);
  var cast1 = bitcast<i32>(precast4);
  var val0 = data4_32[cast1];
  var val1 = data2_32[cast1];
  var val2 = data3_32[cast1];
  var val3 = data5_32[cast1];
  var alu0 = (cast1+1);
  var val4 = data4_32[alu0];
  var val5 = data2_32[alu0];
  var val6 = data3_32[alu0];
  var val7 = data5_32[alu0];
  var alu1 = (cast1+2);
  var val8 = data4_32[alu1];
  var val9 = data2_32[alu1];
  var val10 = data3_32[alu1];
  var val11 = data5_32[alu1];
  var alu2 = (cast1+3);
  var val12 = data4_32[alu2];
  var val13 = data2_32[alu2];
  var val14 = data3_32[alu2];
  var val15 = data5_32[alu2];
  var precast5 = (bitcast<u32>(precast2)<<1u);
  var alu3 = (bitcast<i32>(precast3)+(gidx1*80)+(lidx0*1600)+(lidx1*40)+bitcast<i32>(precast5));
  var val16 = data1_12800[alu3];
  var val17 = data1_12800[(alu3+1)];
  var val18 = data1_12800[(alu3+20)];
  var val19 = data1_12800[(alu3+21)];
  var val20 = data1_12800[(alu3+400)];
  var val21 = data1_12800[(alu3+401)];
  var val22 = data1_12800[(alu3+420)];
  var val23 = data1_12800[(alu3+421)];
  var val24 = data1_12800[(alu3+800)];
  var val25 = data1_12800[(alu3+801)];
  var val26 = data1_12800[(alu3+820)];
  var val27 = data1_12800[(alu3+821)];
  var val28 = data1_12800[(alu3+1200)];
  var val29 = data1_12800[(alu3+1201)];
  var val30 = data1_12800[(alu3+1220)];
  var val31 = data1_12800[(alu3+1221)];
  var alu4 = (1/sqrt((val0+1e-05f)));
  var alu5 = (1/sqrt((val4+1e-05f)));
  var alu6 = (1/sqrt((val8+1e-05f)));
  var alu7 = (1/sqrt((val12+1e-05f)));
  var precast6 = (cast0<<1u);
  var alu8 = (lidx2+bitcast<i32>(precast6)+(gidx1*20)+(lidx0*400)+(lidx1*10));
  var alu9 = ((((val16*(1/(1.0f+exp2((val16*-1.4426950408889634f)))))-val1)*val2*alu4)+val3);
  var alu10 = ((((val17*(1/(1.0f+exp2((val17*-1.4426950408889634f)))))-val1)*val2*alu4)+val3);
  var alu11 = ((((val18*(1/(1.0f+exp2((val18*-1.4426950408889634f)))))-val1)*val2*alu4)+val3);
  var alu12 = ((((val19*(1/(1.0f+exp2((val19*-1.4426950408889634f)))))-val1)*val2*alu4)+val3);
  var alu13 = ((((val20*(1/(1.0f+exp2((val20*-1.4426950408889634f)))))-val5)*val6*alu5)+val7);
  var alu14 = ((((val21*(1/(1.0f+exp2((val21*-1.4426950408889634f)))))-val5)*val6*alu5)+val7);
  var alu15 = ((((val22*(1/(1.0f+exp2((val22*-1.4426950408889634f)))))-val5)*val6*alu5)+val7);
  var alu16 = ((((val23*(1/(1.0f+exp2((val23*-1.4426950408889634f)))))-val5)*val6*alu5)+val7);
  var alu17 = ((((val24*(1/(1.0f+exp2((val24*-1.4426950408889634f)))))-val9)*val10*alu6)+val11);
  var alu18 = ((((val25*(1/(1.0f+exp2((val25*-1.4426950408889634f)))))-val9)*val10*alu6)+val11);
  var alu19 = ((((val26*(1/(1.0f+exp2((val26*-1.4426950408889634f)))))-val9)*val10*alu6)+val11);
  var alu20 = ((((val27*(1/(1.0f+exp2((val27*-1.4426950408889634f)))))-val9)*val10*alu6)+val11);
  var alu21 = ((((val28*(1/(1.0f+exp2((val28*-1.4426950408889634f)))))-val13)*val14*alu7)+val15);
  var alu22 = ((((val29*(1/(1.0f+exp2((val29*-1.4426950408889634f)))))-val13)*val14*alu7)+val15);
  var alu23 = ((((val30*(1/(1.0f+exp2((val30*-1.4426950408889634f)))))-val13)*val14*alu7)+val15);
  var alu24 = ((((val31*(1/(1.0f+exp2((val31*-1.4426950408889634f)))))-val13)*val14*alu7)+val15);
  var alu25 = select(alu9,alu11,(alu9<alu11));
  var alu26 = select(alu13,alu15,(alu13<alu15));
  var alu27 = select(alu17,alu19,(alu17<alu19));
  var alu28 = select(alu21,alu23,(alu21<alu23));
  var alu29 = select(alu25,alu10,(alu25<alu10));
  var alu30 = select(alu26,alu14,(alu26<alu14));
  var alu31 = select(alu27,alu18,(alu27<alu18));
  var alu32 = select(alu28,alu22,(alu28<alu22));
  var alu33 = select(alu29,alu12,(alu29<alu12));
  data0_3200[alu8] = alu33;
  var alu35 = select(alu30,alu16,(alu30<alu16));
  data0_3200[(alu8+100)] = alu35;
  var alu37 = select(alu31,alu20,(alu31<alu20));
  data0_3200[(alu8+200)] = alu37;
  var alu39 = select(alu32,alu24,(alu32<alu24));
  data0_3200[(alu8+300)] = alu39;
}`;

const r_2_8_8_2_4_4_32_3_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_4096:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_3200:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_18432:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_64:array<f32>;
@compute @workgroup_size(8,8,2) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 8 */
  var lidx2 = i32(lindex.z); /* 2 */
  var precast0 = lidx2;
  var precast1 = (bitcast<u32>(precast0)<<2u);
  var cast0 = bitcast<i32>(precast1);
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  acc0[3] = 0.0f;
  acc0[4] = 0.0f;
  acc0[5] = 0.0f;
  acc0[6] = 0.0f;
  acc0[7] = 0.0f;
  acc0[8] = 0.0f;
  acc0[9] = 0.0f;
  acc0[10] = 0.0f;
  acc0[11] = 0.0f;
  acc0[12] = 0.0f;
  acc0[13] = 0.0f;
  acc0[14] = 0.0f;
  acc0[15] = 0.0f;
  for (var ridx1006 = 0; ridx1006 < 32; ridx1006++) {
    var alu16 = ((gidx0*9216)+(lidx0*1152)+(ridx1006*9));
    var val0 = data2_18432[alu16];
    var val1 = data2_18432[(alu16+1)];
    var val2 = data2_18432[(alu16+2)];
    var val3 = data2_18432[(alu16+3)];
    var val4 = data2_18432[(alu16+4)];
    var val5 = data2_18432[(alu16+5)];
    var val6 = data2_18432[(alu16+6)];
    var val7 = data2_18432[(alu16+7)];
    var val8 = data2_18432[(alu16+8)];
    var val9 = data2_18432[(alu16+288)];
    var val10 = data2_18432[(alu16+289)];
    var val11 = data2_18432[(alu16+290)];
    var val12 = data2_18432[(alu16+291)];
    var val13 = data2_18432[(alu16+292)];
    var val14 = data2_18432[(alu16+293)];
    var val15 = data2_18432[(alu16+294)];
    var val16 = data2_18432[(alu16+295)];
    var val17 = data2_18432[(alu16+296)];
    var val18 = data2_18432[(alu16+576)];
    var val19 = data2_18432[(alu16+577)];
    var val20 = data2_18432[(alu16+578)];
    var val21 = data2_18432[(alu16+579)];
    var val22 = data2_18432[(alu16+580)];
    var val23 = data2_18432[(alu16+581)];
    var val24 = data2_18432[(alu16+582)];
    var val25 = data2_18432[(alu16+583)];
    var val26 = data2_18432[(alu16+584)];
    var val27 = data2_18432[(alu16+864)];
    var val28 = data2_18432[(alu16+865)];
    var val29 = data2_18432[(alu16+866)];
    var val30 = data2_18432[(alu16+867)];
    var val31 = data2_18432[(alu16+868)];
    var val32 = data2_18432[(alu16+869)];
    var val33 = data2_18432[(alu16+870)];
    var val34 = data2_18432[(alu16+871)];
    var val35 = data2_18432[(alu16+872)];
    var alu17 = ((lidx1*10)+cast0+(ridx1006*100));
    var val36 = data1_3200[alu17];
    var val37 = data1_3200[(alu17+1)];
    var val38 = data1_3200[(alu17+2)];
    var val39 = data1_3200[(alu17+3)];
    var val40 = data1_3200[(alu17+4)];
    var val41 = data1_3200[(alu17+5)];
    var val42 = data1_3200[(alu17+10)];
    var val43 = data1_3200[(alu17+11)];
    var val44 = data1_3200[(alu17+12)];
    var val45 = data1_3200[(alu17+13)];
    var val46 = data1_3200[(alu17+14)];
    var val47 = data1_3200[(alu17+15)];
    var val48 = data1_3200[(alu17+20)];
    var val49 = data1_3200[(alu17+21)];
    var val50 = data1_3200[(alu17+22)];
    var val51 = data1_3200[(alu17+23)];
    var val52 = data1_3200[(alu17+24)];
    var val53 = data1_3200[(alu17+25)];
    acc0[4] = (acc0[4]+(val37*val0)+(val43*val3)+(val49*val6)+(val38*val1)+(val44*val4)+(val50*val7)+(val39*val2)+(val45*val5)+(val51*val8));
    acc0[5] = (acc0[5]+(val37*val9)+(val43*val12)+(val49*val15)+(val38*val10)+(val44*val13)+(val50*val16)+(val39*val11)+(val45*val14)+(val51*val17));
    acc0[6] = (acc0[6]+(val37*val18)+(val43*val21)+(val49*val24)+(val38*val19)+(val44*val22)+(val50*val25)+(val39*val20)+(val45*val23)+(val51*val26));
    acc0[7] = (acc0[7]+(val37*val27)+(val43*val30)+(val49*val33)+(val38*val28)+(val44*val31)+(val50*val34)+(val39*val29)+(val45*val32)+(val51*val35));
    acc0[8] = (acc0[8]+(val38*val0)+(val44*val3)+(val50*val6)+(val39*val1)+(val45*val4)+(val51*val7)+(val40*val2)+(val46*val5)+(val52*val8));
    acc0[9] = (acc0[9]+(val38*val9)+(val44*val12)+(val50*val15)+(val39*val10)+(val45*val13)+(val51*val16)+(val40*val11)+(val46*val14)+(val52*val17));
    acc0[10] = (acc0[10]+(val38*val18)+(val44*val21)+(val50*val24)+(val39*val19)+(val45*val22)+(val51*val25)+(val40*val20)+(val46*val23)+(val52*val26));
    acc0[11] = (acc0[11]+(val38*val27)+(val44*val30)+(val50*val33)+(val39*val28)+(val45*val31)+(val51*val34)+(val40*val29)+(val46*val32)+(val52*val35));
    acc0[12] = (acc0[12]+(val39*val0)+(val45*val3)+(val51*val6)+(val40*val1)+(val46*val4)+(val52*val7)+(val41*val2)+(val47*val5)+(val53*val8));
    acc0[13] = (acc0[13]+(val39*val9)+(val45*val12)+(val51*val15)+(val40*val10)+(val46*val13)+(val52*val16)+(val41*val11)+(val47*val14)+(val53*val17));
    acc0[14] = (acc0[14]+(val39*val18)+(val45*val21)+(val51*val24)+(val40*val19)+(val46*val22)+(val52*val25)+(val41*val20)+(val47*val23)+(val53*val26));
    acc0[15] = (acc0[15]+(val39*val27)+(val45*val30)+(val51*val33)+(val40*val28)+(val46*val31)+(val52*val34)+(val41*val29)+(val47*val32)+(val53*val35));
    acc0[1] = (acc0[1]+(val36*val9)+(val42*val12)+(val48*val15)+(val37*val10)+(val43*val13)+(val49*val16)+(val38*val11)+(val44*val14)+(val50*val17));
    acc0[2] = (acc0[2]+(val36*val18)+(val42*val21)+(val48*val24)+(val37*val19)+(val43*val22)+(val49*val25)+(val38*val20)+(val44*val23)+(val50*val26));
    acc0[3] = (acc0[3]+(val36*val27)+(val42*val30)+(val48*val33)+(val37*val28)+(val43*val31)+(val49*val34)+(val38*val29)+(val44*val32)+(val50*val35));
    acc0[0] = (acc0[0]+(val36*val0)+(val42*val3)+(val48*val6)+(val37*val1)+(val43*val4)+(val49*val7)+(val38*val2)+(val44*val5)+(val50*val8));
  }
  var precast2 = gidx0;
  var precast3 = lidx0;
  var cast1 = bitcast<u32>(precast2);
  var cast2 = bitcast<u32>(precast3);
  var precast4 = (cast1<<5u);
  var precast5 = (cast2<<2u);
  var alu35 = (bitcast<i32>(precast4)+bitcast<i32>(precast5));
  var val54 = data3_64[alu35];
  var val55 = data3_64[(alu35+1)];
  var val56 = data3_64[(alu35+2)];
  var val57 = data3_64[(alu35+3)];
  var precast6 = lidx1;
  var alu36 = (acc0[0]+val54);
  var alu37 = (acc0[1]+val55);
  var alu38 = (acc0[2]+val56);
  var alu39 = (acc0[3]+val57);
  var alu40 = (acc0[4]+val54);
  var alu41 = (acc0[5]+val55);
  var alu42 = (acc0[6]+val56);
  var alu43 = (acc0[7]+val57);
  var alu44 = (acc0[8]+val54);
  var alu45 = (acc0[9]+val55);
  var alu46 = (acc0[10]+val56);
  var alu47 = (acc0[11]+val57);
  var alu48 = (acc0[12]+val54);
  var alu49 = (acc0[13]+val55);
  var alu50 = (acc0[14]+val56);
  var alu51 = (acc0[15]+val57);
  var precast7 = (cast1<<11u);
  var precast8 = (cast2<<8u);
  var precast9 = (bitcast<u32>(precast6)<<3u);
  var alu52 = (bitcast<i32>(precast7)+bitcast<i32>(precast8)+bitcast<i32>(precast9)+cast0);
  data0_4096[alu52] = (alu36*(1/(1.0f+exp2((alu36*-1.4426950408889634f)))));
  data0_4096[(alu52+1)] = (alu40*(1/(1.0f+exp2((alu40*-1.4426950408889634f)))));
  data0_4096[(alu52+2)] = (alu44*(1/(1.0f+exp2((alu44*-1.4426950408889634f)))));
  data0_4096[(alu52+3)] = (alu48*(1/(1.0f+exp2((alu48*-1.4426950408889634f)))));
  data0_4096[(alu52+64)] = (alu37*(1/(1.0f+exp2((alu37*-1.4426950408889634f)))));
  data0_4096[(alu52+65)] = (alu41*(1/(1.0f+exp2((alu41*-1.4426950408889634f)))));
  data0_4096[(alu52+66)] = (alu45*(1/(1.0f+exp2((alu45*-1.4426950408889634f)))));
  data0_4096[(alu52+67)] = (alu49*(1/(1.0f+exp2((alu49*-1.4426950408889634f)))));
  data0_4096[(alu52+128)] = (alu38*(1/(1.0f+exp2((alu38*-1.4426950408889634f)))));
  data0_4096[(alu52+129)] = (alu42*(1/(1.0f+exp2((alu42*-1.4426950408889634f)))));
  data0_4096[(alu52+130)] = (alu46*(1/(1.0f+exp2((alu46*-1.4426950408889634f)))));
  data0_4096[(alu52+131)] = (alu50*(1/(1.0f+exp2((alu50*-1.4426950408889634f)))));
  data0_4096[(alu52+192)] = (alu39*(1/(1.0f+exp2((alu39*-1.4426950408889634f)))));
  data0_4096[(alu52+193)] = (alu43*(1/(1.0f+exp2((alu43*-1.4426950408889634f)))));
  data0_4096[(alu52+194)] = (alu47*(1/(1.0f+exp2((alu47*-1.4426950408889634f)))));
  data0_4096[(alu52+195)] = (alu51*(1/(1.0f+exp2((alu51*-1.4426950408889634f)))));
}`;

const r_4_2_16_3_2_3_64_3_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_2304:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_4096:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_36864:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_64:array<f32>;
@compute @workgroup_size(16,3,2) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,3>;
  var gidx0 = i32(gindex.x); /* 2 */
  var gidx1 = i32(gindex.y); /* 4 */
  var lidx0 = i32(lindex.x); /* 16 */
  var lidx1 = i32(lindex.y); /* 3 */
  var lidx2 = i32(lindex.z); /* 2 */
  var alu0 = (lidx2*3);
  var precast0 = lidx1;
  acc0[0] = 0.0f;
  acc0[1] = 0.0f;
  acc0[2] = 0.0f;
  var precast1 = (bitcast<u32>(precast0)<<3u);
  for (var ridx1006 = 0; ridx1006 < 64; ridx1006++) {
    var precast2 = ridx1006;
    var alu4 = ((gidx1*9216)+(lidx0*576)+(ridx1006*9));
    var val0 = data2_36864[alu4];
    var val1 = data2_36864[(alu4+1)];
    var val2 = data2_36864[(alu4+2)];
    var val3 = data2_36864[(alu4+3)];
    var val4 = data2_36864[(alu4+4)];
    var val5 = data2_36864[(alu4+5)];
    var val6 = data2_36864[(alu4+6)];
    var val7 = data2_36864[(alu4+7)];
    var val8 = data2_36864[(alu4+8)];
    var precast3 = (bitcast<u32>(precast2)<<6u);
    var alu5 = ((gidx0*24)+bitcast<i32>(precast1)+alu0+bitcast<i32>(precast3));
    var val9 = data1_4096[alu5];
    var val10 = data1_4096[(alu5+1)];
    var val11 = data1_4096[(alu5+2)];
    var val12 = data1_4096[(alu5+3)];
    var val13 = data1_4096[(alu5+4)];
    var val14 = data1_4096[(alu5+8)];
    var val15 = data1_4096[(alu5+9)];
    var val16 = data1_4096[(alu5+10)];
    var val17 = data1_4096[(alu5+11)];
    var val18 = data1_4096[(alu5+12)];
    var val19 = data1_4096[(alu5+16)];
    var val20 = data1_4096[(alu5+17)];
    var val21 = data1_4096[(alu5+18)];
    var val22 = data1_4096[(alu5+19)];
    var val23 = data1_4096[(alu5+20)];
    acc0[1] = (acc0[1]+(val10*val0)+(val15*val3)+(val20*val6)+(val11*val1)+(val16*val4)+(val21*val7)+(val12*val2)+(val17*val5)+(val22*val8));
    acc0[2] = (acc0[2]+(val11*val0)+(val16*val3)+(val21*val6)+(val12*val1)+(val17*val4)+(val22*val7)+(val13*val2)+(val18*val5)+(val23*val8));
    acc0[0] = (acc0[0]+(val9*val0)+(val14*val3)+(val19*val6)+(val10*val1)+(val15*val4)+(val20*val7)+(val11*val2)+(val16*val5)+(val21*val8));
  }
  var precast4 = gidx1;
  var precast5 = (bitcast<u32>(precast4)<<4u);
  var val24 = data3_64[(lidx0+bitcast<i32>(precast5))];
  var alu10 = ((gidx0*18)+(gidx1*576)+(lidx0*36)+(lidx1*6)+alu0);
  data0_2304[alu10] = (acc0[0]+val24);
  data0_2304[(alu10+1)] = (acc0[1]+val24);
  data0_2304[(alu10+2)] = (acc0[2]+val24);
}`;

const r_8_8_3_3_2_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_576:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_2304:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_64:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_64:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_64:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_64:array<f32>;
@compute @workgroup_size(8,3,3) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 3 */
  var lidx2 = i32(lindex.z); /* 3 */
  var precast0 = gidx0;
  var precast1 = lidx2;
  var precast2 = (bitcast<u32>(precast0)<<3u);
  var alu0 = (lidx0+bitcast<i32>(precast2));
  var val0 = data2_64[alu0];
  var val1 = data3_64[alu0];
  var val2 = data4_64[alu0];
  var val3 = data5_64[alu0];
  var precast3 = (bitcast<u32>(precast1)<<1u);
  var alu1 = ((gidx0*288)+(lidx0*36)+(lidx1*12)+bitcast<i32>(precast3));
  var val4 = data1_2304[alu1];
  var val5 = data1_2304[(alu1+1)];
  var val6 = data1_2304[(alu1+6)];
  var val7 = data1_2304[(alu1+7)];
  var alu2 = (1/sqrt((val2+1e-05f)));
  var alu3 = ((((val5*(1/(1.0f+exp2((val5*-1.4426950408889634f)))))-val0)*val1*alu2)+val3);
  var alu4 = ((((val6*(1/(1.0f+exp2((val6*-1.4426950408889634f)))))-val0)*val1*alu2)+val3);
  var alu5 = ((((val7*(1/(1.0f+exp2((val7*-1.4426950408889634f)))))-val0)*val1*alu2)+val3);
  var alu6 = ((((val4*(1/(1.0f+exp2((val4*-1.4426950408889634f)))))-val0)*val1*alu2)+val3);
  var alu7 = select(alu6,alu4,(alu6<alu4));
  var alu8 = select(alu7,alu3,(alu7<alu3));
  var alu9 = select(alu8,alu5,(alu8<alu5));
  data0_576[(lidx2+(gidx0*72)+(lidx0*9)+(lidx1*3))] = alu9;
}`;

const r_10_16_36 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_10:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_576:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_5760:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_10:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var acc0: array<f32,1>;
  var acc1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 10 */
  var lidx0 = i32(lindex.x); /* 16 */
  acc1[0] = 0.0f;
  var val0 = data3_10[gidx0];
  var alu1 = (lidx0*36);
  acc0[0] = 0.0f;
  for (var ridx3002 = 0; ridx3002 < 36; ridx3002++) {
    var val1 = data2_5760[((gidx0*576)+alu1+ridx3002)];
    var val2 = data1_576[(alu1+ridx3002)];
    acc0[0] = (acc0[0]+(val2*val1));
  }
  temp0[lidx0] = acc0[0];
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    for (var ridx1001 = 0; ridx1001 < 16; ridx1001++) {
      var val3 = temp0[ridx1001];
      acc1[0] = (acc1[0]+val3);
    }
    data0_10[gidx0] = (acc1[0]+val0);
  }
}`;

const setupNet = async (device, safetensor) => {
    const metadata = getTensorMetadata(safetensor);
    const infinityBuf = createInfinityUniformBuf(device);

    const layouts=[device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]})]

    const buf_0 = createEmptyBuf(device, 73728);;
    const input0 = createEmptyBuf(device, 3136);;
    const buf_1 = createWeightBuf(device, 3200, getTensorBuffer(safetensor, metadata['layers.0.weight']));
    const buf_2 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['layers.0.bias']));
    const buf_3 = createEmptyBuf(device, 51200);;
    const buf_4 = createWeightBuf(device, 102400, getTensorBuffer(safetensor, metadata['layers.2.weight']));
    const buf_5 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['layers.2.bias']));
    const buf_6 = createEmptyBuf(device, 12800);;
    const buf_7 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['layers.4.running_mean']));
    const buf_8 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['layers.4.weight']));
    const buf_9 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['layers.4.running_var']));
    const buf_10 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['layers.4.bias']));
    const buf_11 = createEmptyBuf(device, 16384);;
    const buf_12 = createWeightBuf(device, 73728, getTensorBuffer(safetensor, metadata['layers.6.weight']));
    const buf_13 = createWeightBuf(device, 256, getTensorBuffer(safetensor, metadata['layers.6.bias']));
    const buf_14 = createEmptyBuf(device, 9216);;
    const buf_15 = createWeightBuf(device, 147456, getTensorBuffer(safetensor, metadata['layers.8.weight']));
    const buf_16 = createWeightBuf(device, 256, getTensorBuffer(safetensor, metadata['layers.8.bias']));
    const buf_17 = createEmptyBuf(device, 2304);;
    const buf_18 = createWeightBuf(device, 256, getTensorBuffer(safetensor, metadata['layers.10.running_mean']));
    const buf_19 = createWeightBuf(device, 256, getTensorBuffer(safetensor, metadata['layers.10.weight']));
    const buf_20 = createWeightBuf(device, 256, getTensorBuffer(safetensor, metadata['layers.10.running_var']));
    const buf_21 = createWeightBuf(device, 256, getTensorBuffer(safetensor, metadata['layers.10.bias']));
    const output0 = createEmptyBuf(device, 40);;
    const buf_22 = createWeightBuf(device, 23040, getTensorBuffer(safetensor, metadata['layers.13.weight']));
    const buf_23 = createWeightBuf(device, 40, getTensorBuffer(safetensor, metadata['layers.13.bias']));

    const gpuWriteBuffer0 = device.createBuffer({size:input0.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });

    const gpuReadBuffer0 = device.createBuffer({size:output0.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const kernels = [r_4_3_2_8_8_3_4_5_5, r_5_5_8_4_4_4_32_5_5, r_5_5_8_2_2_4_2_2, r_2_8_8_2_4_4_32_3_3, r_4_2_16_3_2_3_64_3_3, r_8_8_3_3_2_2, r_10_16_36];
    const pipelines = await Promise.all(kernels.map(async (name, i) => {
      return await device.createComputePipelineAsync({
          layout: device.createPipelineLayout({
              bindGroupLayouts: [layouts[i]],
          }),
          compute: {
              module: device.createShaderModule({
                  code: name,
              }),
              entryPoint: "main",
          },
      });
  }))

    return async (_input0) => {
        const commandEncoder = device.createCommandEncoder();
        await gpuWriteBuffer0.mapAsync(GPUMapMode.WRITE);
        new Float32Array(gpuWriteBuffer0.getMappedRange()).set(_input0);
        gpuWriteBuffer0.unmap();
        commandEncoder.copyBufferToBuffer(gpuWriteBuffer0, 0, input0, 0, gpuWriteBuffer0.size);
        addComputePass(device, commandEncoder, pipelines[0], layouts[0], infinityBuf, [buf_0, input0, buf_1, buf_2], [3, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[1], layouts[1], infinityBuf, [buf_3, buf_0, buf_4, buf_5], [5, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[2], layouts[2], infinityBuf, [buf_6, buf_3, buf_7, buf_8, buf_9, buf_10], [5, 5, 1]);
        addComputePass(device, commandEncoder, pipelines[3], layouts[3], infinityBuf, [buf_11, buf_6, buf_12, buf_13], [2, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[4], layouts[4], infinityBuf, [buf_14, buf_11, buf_15, buf_16], [2, 4, 1]);
        addComputePass(device, commandEncoder, pipelines[5], layouts[5], infinityBuf, [buf_17, buf_14, buf_18, buf_19, buf_20, buf_21], [8, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[6], layouts[6], infinityBuf, [output0, buf_17, buf_22, buf_23], [10, 1, 1]);
        commandEncoder.copyBufferToBuffer(output0, 0, gpuReadBuffer0, 0, output0.size);
        const gpuCommands = commandEncoder.finish();
        device.queue.submit([gpuCommands]);

        await gpuReadBuffer0.mapAsync(GPUMapMode.READ);
        const resultBuffer0 = new Float32Array(gpuReadBuffer0.size/4);
        resultBuffer0.set(new Float32Array(gpuReadBuffer0.getMappedRange()));
        gpuReadBuffer0.unmap();
        return [resultBuffer0];
    }
}
const load = async (device, weight_path) => { return await fetch(weight_path).then(x => x.arrayBuffer()).then(x => setupNet(device, new Uint8Array(x))); }
return { load, setupNet };
})();
export default mnist_convnet;
