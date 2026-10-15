'use strict';
const video=document.getElementById('filmVideo');
const chapterButtons=[...document.querySelectorAll('[data-time]')];
chapterButtons.forEach(button=>button.addEventListener('click',()=>{video.currentTime=Number(button.dataset.time);video.play().catch(()=>{});}));
video.addEventListener('timeupdate',()=>{let current=-1;for(let i=chapterButtons.length-1;i>=0;i--){if(video.currentTime>=Number(chapterButtons[i].dataset.time)){current=i;break;}}chapterButtons.forEach((b,i)=>b.classList.toggle('active',i===current));});
const motionPreference=window.matchMedia('(prefers-reduced-motion: reduce)');
const motionControllers=[];
document.querySelectorAll('.gif-toggle').forEach(button=>{const img=document.getElementById(button.dataset.image);let moving=!motionPreference.matches;function update(){img.src=moving?img.dataset.motion:img.dataset.still;button.textContent=moving?'暂停 GIF':'播放 GIF';button.setAttribute('aria-label',button.textContent+'：'+img.alt);button.setAttribute('aria-pressed',String(moving));}button.disabled=false;update();button.addEventListener('click',()=>{moving=!moving;update();});motionControllers.push(()=>{moving=!motionPreference.matches;update();});});
function syncMotionPreference(){motionControllers.forEach(update=>update());}
if(motionPreference.addEventListener)motionPreference.addEventListener('change',syncMotionPreference);else if(motionPreference.addListener)motionPreference.addListener(syncMotionPreference);
