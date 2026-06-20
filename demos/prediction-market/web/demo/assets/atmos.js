/* VERITAS — shared atmosphere + reveal + parallax.
   Safe to include on any page; every block guards for its element. */
(function(){
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---- scroll reveal ---- */
  var reveals = document.querySelectorAll('.reveal');
  if(reveals.length){
    if('IntersectionObserver' in window){
      var io = new IntersectionObserver(function(entries){
        entries.forEach(function(e){ if(e.isIntersecting){ e.target.classList.add('in'); io.unobserve(e.target); } });
      }, {threshold:0.18, rootMargin:'0px 0px -8% 0px'});
      reveals.forEach(function(el){ io.observe(el); });
    } else {
      reveals.forEach(function(el){ el.classList.add('in'); });
    }
  }

  /* ---- particle starfield ---- */
  var c = document.getElementById('stars');
  if(c && c.getContext){
    var ctx = c.getContext('2d'), dpr = Math.min(window.devicePixelRatio||1, 2);
    var parts = [], W, H, raf;
    function size(){
      W = c.clientWidth; H = c.clientHeight;
      c.width = W*dpr; c.height = H*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
      var n = Math.round((W*H)/26000);
      parts = [];
      for(var i=0;i<n;i++){
        parts.push({x:Math.random()*W,y:Math.random()*H,r:Math.random()*1.3+0.3,
          vy:(Math.random()*0.18+0.04),vx:(Math.random()-0.5)*0.08,a:Math.random()*0.5+0.15,tw:Math.random()*Math.PI*2});
      }
    }
    function draw(){
      ctx.clearRect(0,0,W,H);
      for(var i=0;i<parts.length;i++){
        var p = parts[i];
        p.y += p.vy; p.x += p.vx; p.tw += 0.02;
        if(p.y>H+4){p.y=-4;p.x=Math.random()*W;}
        if(p.x<-4)p.x=W+4; if(p.x>W+4)p.x=-4;
        var a = p.a * (0.6 + 0.4*Math.sin(p.tw));
        ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
        ctx.fillStyle = 'rgba(120,225,210,'+a.toFixed(3)+')'; ctx.fill();
      }
      raf = requestAnimationFrame(draw);
    }
    if(!reduce){
      size(); draw();
      var rt; window.addEventListener('resize', function(){ clearTimeout(rt); rt=setTimeout(size,180); });
    } else {
      size();
      for(var i=0;i<parts.length;i++){var p=parts[i];ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);ctx.fillStyle='rgba(120,225,210,'+(p.a*0.6).toFixed(3)+')';ctx.fill();}
    }
  }

  /* ---- parallax on any [data-parallax] element (e.g. the hero seal) ---- */
  if(!reduce){
    var px = document.querySelectorAll('[data-parallax]');
    if(px.length){
      window.addEventListener('mousemove', function(e){
        var dx = (e.clientX/window.innerWidth - 0.5), dy = (e.clientY/window.innerHeight - 0.5);
        px.forEach(function(el){
          var k = parseFloat(el.getAttribute('data-parallax')) || 14;
          el.style.transform = 'translate('+(dx*k).toFixed(2)+'px,'+(dy*k*0.85).toFixed(2)+'px)';
        });
      }, {passive:true});
    }
  }

  /* ---- proof-ticket cinematic sequence (index hero only) ---- */
  var ticket = document.querySelector('.ticket');
  if(ticket){
    var tSteps = ticket.querySelectorAll('.tk-step');
    var tVerdict = document.getElementById('tkVerdict');
    var tChip = document.getElementById('tkChip');
    var tSlash = document.getElementById('tkSlash');
    var on = function(el){ if(el) el.classList.add('in'); };
    if(reduce){
      tSteps.forEach(function(s){ s.classList.add('in','done'); });
      on(tVerdict); on(tChip); if(tSlash) tSlash.classList.add('fire');
    } else {
      var base = 620;
      setTimeout(function(){ on(tChip); }, base);
      tSteps.forEach(function(s,i){
        setTimeout(function(){ on(s); }, base+220+i*560);
        setTimeout(function(){ s.classList.add('done'); }, base+220+i*560+340);
      });
      var vT = base+220+tSteps.length*560+150;
      setTimeout(function(){ on(tVerdict); }, vT);
      setTimeout(function(){ if(tSlash) tSlash.classList.add('fire'); }, vT+560);
    }
  }
})();
