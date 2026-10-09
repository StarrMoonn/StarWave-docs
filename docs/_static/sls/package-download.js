/* Bounded-part, same-origin ZIP download with incremental SHA-256 verification.
   No external scripts, credentials, cookies, uploads or third-party services. */
(function (root) {
  'use strict';
  const K = new Uint32Array([0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2]);
  const rr = (x,n) => (x>>>n)|(x<<(32-n));
  class SHA256 {
    constructor(){this.h=new Uint32Array([0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19]);this.tail=new Uint8Array(64);this.used=0;this.bytes=0;this.w=new Uint32Array(64);}
    block(data,offset){const w=this.w;for(let i=0;i<16;i++){const j=offset+4*i;w[i]=(data[j]<<24)|(data[j+1]<<16)|(data[j+2]<<8)|data[j+3];}for(let i=16;i<64;i++){const x=w[i-15],y=w[i-2];w[i]=(w[i-16]+(rr(x,7)^rr(x,18)^(x>>>3))+w[i-7]+(rr(y,17)^rr(y,19)^(y>>>10)))>>>0;}let [a,b,c,d,e,f,g,h]=this.h;for(let i=0;i<64;i++){const t1=(h+(rr(e,6)^rr(e,11)^rr(e,25))+((e&f)^(~e&g))+K[i]+w[i])>>>0;const t2=((rr(a,2)^rr(a,13)^rr(a,22))+((a&b)^(a&c)^(b&c)))>>>0;h=g;g=f;f=e;e=(d+t1)>>>0;d=c;c=b;b=a;a=(t1+t2)>>>0;}[a,b,c,d,e,f,g,h].forEach((x,i)=>{this.h[i]=(this.h[i]+x)>>>0;});}
    update(data){if(!(data instanceof Uint8Array))data=new Uint8Array(data);this.bytes+=data.length;let offset=0;if(this.used){const n=Math.min(64-this.used,data.length);this.tail.set(data.subarray(0,n),this.used);this.used+=n;offset=n;if(this.used===64){this.block(this.tail,0);this.used=0;}}while(offset+64<=data.length){this.block(data,offset);offset+=64;}if(offset<data.length){this.tail.set(data.subarray(offset),0);this.used=data.length-offset;}return this;}
    hex(){const pad=new Uint8Array(this.used<56?64:128);pad.set(this.tail.subarray(0,this.used));pad[this.used]=128;const view=new DataView(pad.buffer);view.setUint32(pad.length-8,Math.floor(this.bytes/0x20000000));view.setUint32(pad.length-4,(this.bytes*8)>>>0);for(let i=0;i<pad.length;i+=64)this.block(pad,i);return Array.from(this.h,x=>x.toString(16).padStart(8,'0')).join('');}
  }
  if(typeof module!=='undefined'&&module.exports){module.exports=SHA256;return;}
  root.StarWaveSHA256=SHA256;
  document.querySelectorAll('[data-sw-package-download]').forEach(button=>{
    const status=document.getElementById(button.dataset.status),cancel=document.getElementById(button.dataset.cancel);
    const zh=document.documentElement.lang.startsWith('zh');let busy=false,controller=null,state=null;
    cancel.addEventListener('click',()=>{if(controller)controller.abort();});
    button.addEventListener('click',async()=>{
      if(busy)return;busy=true;button.disabled=true;cancel.hidden=false;controller=new AbortController();
      try{
        const manifestURL=new URL(button.dataset.manifest,location.href);
        if(manifestURL.origin!==location.origin)throw new Error('Manifest must be same-origin');
        status.textContent=zh?'读取下载清单…':'Reading download manifest…';
        const response=await fetch(manifestURL,{signal:controller.signal,credentials:'omit'});if(!response.ok)throw new Error('Manifest HTTP '+response.status);const m=await response.json();
        if(!Array.isArray(m.parts)||m.parts.length>100||m.bytes>100*1024*1024||!/^[-a-zA-Z0-9_.]+\.zip$/.test(m.filename)||!/^[0-9a-f]{64}$/.test(m.sha256))throw new Error('Invalid package manifest');
        if(!state||state.manifest.sha256!==m.sha256)state={manifest:m,index:0,bytes:0,blobs:[],hash:new SHA256()};
        for(;state.index<m.parts.length;state.index++){
          const part=m.parts[state.index];if(!/^[a-zA-Z0-9_.-]+$/.test(part.file)||part.bytes>4*1024*1024||!/^[0-9a-f]{64}$/.test(part.sha256))throw new Error('Invalid part');
          const url=new URL(part.file,manifestURL);let data;
          for(let attempt=0;attempt<3;attempt++){
            status.textContent=(zh?'下载并校验 ':'Downloading and verifying ')+(state.index+1)+'/'+m.parts.length+' · '+(state.bytes/1048576).toFixed(1)+'/'+(m.bytes/1048576).toFixed(1)+' MiB'+(attempt?(zh?' · 重试':' · retry'):'');
            try{const r=await fetch(url,{signal:controller.signal,credentials:'omit'});if(!r.ok)throw new Error('HTTP '+r.status);data=new Uint8Array(await r.arrayBuffer());if(data.length!==part.bytes||new SHA256().update(data).hex()!==part.sha256)throw new Error('Part integrity mismatch');break;}
            catch(error){if(error.name==='AbortError'||attempt===2)throw error;await new Promise(resolve=>setTimeout(resolve,500*(attempt+1)));}
          }
          state.hash.update(data);state.bytes+=data.length;state.blobs.push(new Blob([data]));
          await new Promise(resolve=>setTimeout(resolve,0));
        }
        if(state.bytes!==m.bytes||(state.digest||(state.digest=state.hash.hex()))!==m.sha256){state=null;throw new Error('Complete ZIP integrity mismatch');}
        const blob=new Blob(state.blobs,{type:'application/zip'});const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=m.filename;document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),60000);
        status.textContent=(zh?'完整 ZIP 已校验并交给浏览器下载。SHA-256：':'Complete ZIP verified and handed to your browser. SHA-256: ')+m.sha256;state=null;
      }catch(error){status.textContent=(error.name==='AbortError'?(zh?'下载已暂停。点击按钮可继续。':'Download paused. Click the button to resume.'):(zh?'下载未完成，可点击按钮重试。':'Download incomplete. Click the button to retry.'))+' '+(error.name==='AbortError'?'':error.message);}
      finally{busy=false;button.disabled=false;cancel.hidden=true;controller=null;}
    });
  });
})(typeof window!=='undefined'?window:globalThis);
