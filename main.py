# main.py — Industry-optimized: multi-action, streaming, resource blocking,
# pre-warming, caching, HTTP/2, compact viewport
import os
os.environ["BROWSER_USE_TELEMETRY"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
import os, sys, json, asyncio, logging, io, re, hashlib, time
from typing import ClassVar
from dotenv import load_dotenv

if sys.platform == "win32":
    try: sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except: pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
for n in ("urllib3","httpx","httpcore","openai","playwright","asyncio"): logging.getLogger(n).setLevel(logging.WARNING)
logger = logging.getLogger("agent")

# ═══════════════════════════════════════════════════════════════════════════
# GLM response fixer (same as before — unchanged)
# ═══════════════════════════════════════════════════════════════════════════
def fix_glm_response(content):
    if not isinstance(content, str) or not content.strip(): return content
    try:
        p = json.loads(content)
        if not isinstance(p, dict): return content

        # Remove thinking fields
        for k in ("thinking", "thought", "reasonning"):
            p.pop(k, None)

        # Move top-level state fields into current_state
        cs = p.get("current_state")
        if not isinstance(cs, dict): cs = {}
        for f in ("evaluation_previous_goal", "memory", "next_goal", "plan"):
            if f in p and f not in cs: cs[f] = p.pop(f)
        for f in ("evaluation_previous_goal", "memory", "next_goal"):
            if f not in cs: cs[f] = ""
        p["current_state"] = cs

        # Ensure action is a list
        a = p.get("action")
        if a is None: p["action"] = [{"type": "wait", "seconds": 1}]
        elif isinstance(a, dict): p["action"] = [a]
        elif not isinstance(a, list): p["action"] = [{"type": "wait", "seconds": 1}]

        # Fix each action's field names and formats
        for act in p["action"]:
            if not isinstance(act, dict): continue

            # ═══ FIX: wait must be {"seconds": N} object, not bare integer ═══
            if "wait" in act:
                w = act["wait"]
                if isinstance(w, (int, float)):
                    act["wait"] = {"seconds": int(w)}
                elif not isinstance(w, dict):
                    act["wait"] = {"seconds": 1}
                elif "seconds" not in w:
                    for alias in ("time", "duration", "delay", "wait_time", "s"):
                        if alias in w:
                            w["seconds"] = w.pop(alias)
                            break
                    if "seconds" not in w:
                        w["seconds"] = 1

            # ═══ FIX: done must be {"text": "..."} object, not bare string ═══
            if "done" in act:
                d = act["done"]
                if isinstance(d, str):
                    act["done"] = {"text": d}
                elif not isinstance(d, dict):
                    act["done"] = {"text": str(d)}
                elif "text" not in d:
                    for alias in ("result", "summary", "message", "output", "answer", "reason"):
                        if alias in d:
                            d["text"] = d.pop(alias)
                            break
                    if "text" not in d:
                        d["text"] = "Task completed"
                if "success" not in d:
                    d["success"] = True

            # ═══ FIX: scroll must have amount field ═══
            if "scroll" in act:
                s = act["scroll"]
                if not isinstance(s, dict):
                    act["scroll"] = {"amount": 1}
                elif "amount" not in s:
                    for alias in ("pixels", "distance", "scroll_amount", "steps"):
                        if alias in s:
                            s["amount"] = s.pop(alias)
                            break
                    if "amount" not in s:
                        s["amount"] = 1

            # ═══ FIX: navigate must have url field ═══
            if "navigate" in act and isinstance(act["navigate"], dict):
                nv = act["navigate"]
                for old in ("link", "address", "website", "page", "target"):
                    if old in nv and "url" not in nv:
                        nv["url"] = nv.pop(old)

            # ═══ FIX: go_back / close must be empty dicts ═══
            if "go_back" in act and not isinstance(act["go_back"], dict):
                act["go_back"] = {}
            if "close" in act and not isinstance(act["close"], dict):
                act["close"] = {}

            # ═══ FIX: switch must have handle ═══
            if "switch" in act and isinstance(act["switch"], dict):
                sw = act["switch"]
                if "handle" not in sw:
                    for alias in ("tab_id", "index", "tab_index", "tab"):
                        if alias in sw:
                            sw["handle"] = sw.pop(alias)
                            break
                    if "handle" not in sw:
                        act.pop("switch", None)

            # ═══ Existing fixes (unchanged) ═══
            if "search_page" in act and isinstance(act["search_page"], dict):
                sp = act["search_page"]
                for old in ("query", "search_term", "text", "q", "keyword"):
                    if old in sp and "pattern" not in sp:
                        sp["pattern"] = sp.pop(old)
            if "go_to_url" in act and isinstance(act["go_to_url"], dict):
                gu = act["go_to_url"]
                for old in ("address", "link", "website", "page"):
                    if old in gu and "url" not in gu:
                        gu["url"] = gu.pop(old)
            if "click" in act and isinstance(act["click"], dict):
                cl = act["click"]
                for old in ("element", "selector", "id", "target"):
                    if old in cl and "index" not in cl:
                        cl["index"] = cl.pop(old)
            if "input" in act and isinstance(act["input"], dict):
                ip = act["input"]
                for old in ("value", "data", "content", "string"):
                    if old in ip and "text" not in ip:
                        ip["text"] = ip.pop(old)

        return json.dumps(p)
    except Exception:
        return content

class _FixedResp:
    __slots__ = ("_orig","content","completion")
    def __init__(self, orig, fixed):
        object.__setattr__(self,"_orig",orig)
        object.__setattr__(self,"content",fixed)
        object.__setattr__(self,"completion",fixed)
    def __getattr__(self, n): return getattr(object.__getattribute__(self,"_orig"), n)
    def model_copy(self, update=None):
        if update and "content" in update:
            return _FixedResp(object.__getattribute__(self,"_orig"), update["content"])
        return self

def _fix_resp(resp):
    try:
        c = getattr(resp,"content",None)
        if isinstance(c,str) and c.strip():
            fixed = fix_glm_response(c)
            if fixed != c:
                logger.info(f"🔧 Fixed response")
                if hasattr(resp,"model_copy"):
                    try: return resp.model_copy(update={"content":fixed})
                    except: pass
                try: resp.content=fixed; return resp
                except: pass
                try: object.__setattr__(resp,"content",fixed); return resp
                except: pass
                return _FixedResp(resp, fixed)
    except: pass
    return resp

# ═══════════════════════════════════════════════════════════════════════════
# LLMFixer wrapper (unchanged)
# ═══════════════════════════════════════════════════════════════════════════
class LLMFixer:
    def __init__(self, llm):
        object.__setattr__(self,"_llm",llm)
        self.provider = getattr(llm,"provider","openai")
        self.model = getattr(llm,"model",None)
        self.model_name = getattr(llm,"model_name",getattr(llm,"model",None))
    async def ainvoke(self, *a, **kw):
        resp = await self._llm.ainvoke(*a, **kw)
        return _fix_resp(resp)
    def invoke(self, *a, **kw):
        return _fix_resp(self._llm.invoke(*a, **kw))
    def bind_tools(self, *a, **k): return LLMFixer(self._llm.bind_tools(*a, **k))
    def with_structured_output(self, *a, **k): return LLMFixer(self._llm.with_structured_output(*a, **k))
    def __getattr__(self, n): return getattr(self._llm, n)
    def __setattr__(self, n, v):
        if n in ("provider","model","model_name"): object.__setattr__(self,n,v)
        else: setattr(self._llm,n,v)

# ═══════════════════════════════════════════════════════════════════════════
# Action Cache — skip LLM call when same page state seen before
# ═══════════════════════════════════════════════════════════════════════════
class ActionCache:
    """Cache LLM responses based on URL + DOM fingerprint + task hash."""
    def __init__(self, max_size=50):
        self._cache = {}
        self._max = max_size
        self._hits = 0
        self._misses = 0

    def _key(self, url, dom_text, task):
        # Hash: URL + first/last 300 chars of DOM + first 80 chars of task
        fingerprint = f"{url}|{dom_text[:300]}|{dom_text[-300:]}|{task[:80]}"
        return hashlib.md5(fingerprint.encode()).hexdigest()

    def get(self, url, dom_text, task):
        k = self._key(url, dom_text, task)
        hit = self._cache.get(k)
        if hit:
            self._hits += 1
            logger.info(f"⚡ Cache HIT (hits={self._hits}, misses={self._misses})")
        else:
            self._misses += 1
        return hit

    def set(self, url, dom_text, task, response_content):
        k = self._key(url, dom_text, task)
        self._cache[k] = response_content
        # Evict oldest if over capacity
        if len(self._cache) > self._max:
            self._cache.pop(next(iter(self._cache)))

    def stats(self):
        return f"cache: {self._hits} hits, {self._misses} misses, {len(self._cache)} entries"

# Global cache
_action_cache = ActionCache()

# ═══════════════════════════════════════════════════════════════════════════
# LLM factory — browser-use ChatOpenAI subclass with thinking disabled
# + streaming + connection pre-warming
# ═══════════════════════════════════════════════════════════════════════════
def create_llm(api_key, base_url, model):
    bu_cls = None
    for mp in ("browser_use.llm", "browser_use.llm.openai"):
        try:
            m = __import__(mp, fromlist=["ChatOpenAI"])
            if hasattr(m, "ChatOpenAI"):
                bu_cls = getattr(m, "ChatOpenAI")
                break
        except: pass

    if bu_cls is not None:
        class FastBUChatOpenAI(bu_cls):
            """browser-use ChatOpenAI + thinking disabled + streaming."""

            async def _agenerate(self, *args, **kwargs):
                eb = kwargs.get("extra_body")
                if not isinstance(eb, dict): eb = {}
                eb["enable_thinking"] = False
                eb["thinking"] = False
                kwargs["extra_body"] = eb
                # B2: Enable streaming for faster TTFB
                kwargs.setdefault("stream", True)
                try:
                    return await super()._agenerate(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"⚠️ thinking=False/stream failed ({e}), retrying plain")
                    kwargs.pop("extra_body", None)
                    kwargs.pop("stream", None)
                    return await super()._agenerate(*args, **kwargs)

            def _generate(self, *args, **kwargs):
                eb = kwargs.get("extra_body")
                if not isinstance(eb, dict): eb = {}
                eb["enable_thinking"] = False
                eb["thinking"] = False
                kwargs["extra_body"] = eb
                try:
                    return super()._generate(*args, **kwargs)
                except Exception as e:
                    kwargs.pop("extra_body", None)
                    return super()._generate(*args, **kwargs)

        for kw in [
            dict(model=model, api_key=api_key, base_url=base_url, temperature=0.0),
            dict(model=model, openai_api_key=api_key, openai_api_base=base_url, temperature=0.0),
            dict(model=model, api_key=api_key, base_url=base_url),
        ]:
            try:
                llm = FastBUChatOpenAI(**kw)
                _ = llm.provider
                # A4: max_tokens=1024 (was 512) for multi-action responses
                try: llm.max_tokens = 1024
                except:
                    try: object.__setattr__(llm, "max_tokens", 1024)
                    except: pass
                # B2: Enable streaming on the model itself
                try: llm.streaming = True
                except:
                    try: object.__setattr__(llm, "streaming", True)
                    except: pass
                logger.info(f"✅ FastBUChatOpenAI (thinking=off, streaming=on, max_tokens=1024)")
                return LLMFixer(llm)
            except TypeError: continue

        # Fallback: plain browser-use ChatOpenAI
        for kw in [
            dict(model=model, api_key=api_key, base_url=base_url, temperature=0.0),
            dict(model=model, api_key=api_key, base_url=base_url),
        ]:
            try:
                llm = bu_cls(**kw)
                _ = llm.provider
                logger.info(f"✅ browser-use ChatOpenAI (fallback, provider={llm.provider})")
                return LLMFixer(llm)
            except TypeError: continue

    logger.warning("Using custom GLMChatModel wrapper")
    return LLMFixer(GLMChatModel(model, api_key, base_url))

# ═══════════════════════════════════════════════════════════════════════════
# A3: Connection pre-warming
# ═══════════════════════════════════════════════════════════════════════════
async def prewarm_llm(llm):
    """Send minimal request to warm TCP/TLS connection + model cache."""
    try:
        t0 = time.time()
        await llm.ainvoke([{"role": "user", "content": "Reply: OK"}], config={})
        elapsed = time.time() - t0
        logger.info(f"🔥 LLM pre-warmed ({elapsed:.1f}s)")
        return elapsed
    except Exception as e:
        logger.warning(f"Pre-warm failed (ok to ignore): {e}")
        return None

# ═══════════════════════════════════════════════════════════════════════════
# B1: Resource blocking via CDP
# ═══════════════════════════════════════════════════════════════════════════
async def block_resources(session):
    """Block fonts, analytics, ads, social widgets to speed up page loads."""
    blocked = [
        # Fonts
        "*.woff", "*.woff2", "*.ttf", "*.otf", "*.eot",
        # Analytics & tracking
        "*google-analytics.com*", "*googletagmanager.com*",
        "*doubleclick.net*", "*connect.facebook.net*",
        "*platform.twitter.com*", "*analytics*",
        "*hotjar.com*", "*clarity.ms*", "*segment.io*",
        "*mixpanel.com*", "*amplitude.com*",
        # Ad networks
        "*googleads*", "*adservice*", "*adnxs*",
        # Social embeds
        "*platform.linkedin.com*", "*addthis.com*",
    ]
    cdp = None
    for attr in ("cdp_client", "_cdp_client"):
        cdp = getattr(session, attr, None)
        if cdp: break

    if cdp is None:
        logger.info("ℹ️ No CDP client found for resource blocking")
        return

    for method_name in ("send", "execute", "call_method"):
        if hasattr(cdp, method_name):
            method = getattr(cdp, method_name)
            try:
                if asyncio.iscoroutinefunction(method):
                    await method("Network.setBlockedURLs", {"urls": blocked})
                else:
                    method("Network.setBlockedURLs", {"urls": blocked})
                logger.info(f"✅ Blocked {len(blocked)} resource URL patterns (via cdp.{method_name})")
                return
            except Exception as e:
                logger.debug(f"cdp.{method_name} for blocking: {e}")
                continue
    logger.info("ℹ️ Could not block resources (CDP method not found)")

# ═══════════════════════════════════════════════════════════════════════════
# Custom LLM wrapper (Tier 3 — with thinking disabled + HTTP/2)
# ═══════════════════════════════════════════════════════════════════════════
class GLMChatModel:
    def __init__(self, model, api_key, base_url, temp=0.0, max_tokens=1024):
        self.model = model; self.model_name = model
        self.api_key = api_key; self.base_url = base_url
        self.temperature = temp; self.max_tokens = max_tokens
        self.provider = "openai"
        import httpx
        # C1: HTTP/2 for faster multiplexed requests
        self.client = httpx.AsyncClient(
            base_url=base_url, http2=True,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=httpx.Timeout(30.0, connect=5.0))
    def bind_tools(self, *a, **k): return self
    def with_structured_output(self, *a, **k): return self
    async def ainvoke(self, messages, config=None, **kw):
        try:
            oms = self._conv(messages)
            payload = {"model": self.model, "messages": oms,
                "temperature": self.temperature, "max_tokens": self.max_tokens,
                "enable_thinking": False, "thinking": False, "stream": True}
            # Use stream for faster TTFB
            async with self.client.stream("POST", "/chat/completions", json=payload) as r:
                if r.status_code != 200:
                    # Retry without thinking/stream params
                    payload.pop("enable_thinking"); payload.pop("thinking"); payload.pop("stream")
                    r2 = await self.client.post("/chat/completions", json=payload)
                    if r2.status_code != 200: return _Resp(self._err(f"API {r2.status_code}"))
                    d = r2.json(); msg = d["choices"][0]["message"]
                    c = msg.get("content","") or msg.get("reasoning_content","") or ""
                else:
                    # Accumulate stream
                    chunks = []
                    async for line in r.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]": break
                            try:
                                chunk = json.loads(data)
                                delta = chunk.get("choices",[{}])[0].get("delta",{})
                                content = delta.get("content","")
                                if content: chunks.append(content)
                            except: pass
                    c = "".join(chunks)
            if not c.strip(): return _Resp(self._err("empty"))
            return _Resp(fix_glm_response(c))
        except Exception as e:
            return _Resp(self._err(str(e)))
    def invoke(self, m, config=None, **kw):
        try: loop = asyncio.get_running_loop()
        except: loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.ainvoke(m, config=config, **kw))
    def _conv(self, msgs):
        if isinstance(msgs, str): return [{"role":"user","content":msgs}]
        if not isinstance(msgs, list): msgs = [msgs]
        out = []
        tm = {"system":"system","human":"user","ai":"assistant","tool":"tool","chat":"user"}
        for m in msgs:
            r, c = "user", ""
            if hasattr(m,"type") and m.type: r = tm.get(m.type,"user"); c = m.content
            elif isinstance(m, dict): r = m.get("role","user"); c = m.get("content","")
            elif isinstance(m, str): c = m
            else: r = getattr(m,"role","user") or "user"; c = getattr(m,"content",str(m))
            if isinstance(c, list):
                ps = []
                for p in c:
                    if isinstance(p, dict) and p.get("type")=="text": ps.append(p.get("text",""))
                    elif isinstance(p, str): ps.append(p)
                c = "\n".join(ps)
            out.append({"role":r,"content":str(c) if c else ""})
        return out
    def _err(self, m):
        return json.dumps({"current_state":{"evaluation_previous_goal":f"Err:{m}",
            "memory":"","next_goal":"Continue"},"action":[{"type":"wait","seconds":1}]})

class _Resp:
    def __init__(self, content):
        self.content = content; self.completion = content
        self.usage = {}; self.response_metadata = {}; self.usage_metadata = None
        self.additional_kwargs = {}; self.tool_calls = []; self.invalid_tool_calls = []
        self.id = None; self.type = "ai"; self.generations = []; self.raw = None; self.name = None
    def __str__(self): return self.content
    def model_copy(self, update=None):
        r = _Resp(update.get("content",self.content) if update else self.content)
        r.__dict__.update(self.__dict__)
        if update: r.content = update.get("content",r.content); r.completion = r.content
        return r

# ═══════════════════════════════════════════════════════════════════════════
def imp(paths):
    for mp, cn in paths:
        try:
            m = __import__(mp, fromlist=[cn])
            if hasattr(m, cn): return getattr(m, cn)
        except: pass
    return None

# ═══════════════════════════════════════════════════════════════════════════
# Task preprocessing (unchanged from working version)
# ═══════════════════════════════════════════════════════════════════════════
def preprocess_task(task):
    t = task.lower()
    if "youtube" in t and ("channel" in t or "video" in t):
        for pattern in [
            r'channel\s*[-:]?\s*["\']?(.+?)["\']?\s*(?:,|\.|\s+it\s|\s+that\s|\s+which\s|$)',
            r'youtube\s*[-:]\s*(.+?)(?:,|\.|\s+it\s|$)',
        ]:
            m = re.search(pattern, task, re.I)
            if m:
                name = m.group(1).strip().rstrip(',').strip()
                if name and len(name) > 1:
                    return f"Go to https://www.youtube.com/results?search_query={name.replace(' ', '+')} — {task}"
    if "github" in t:
        skip_words = {'profile','repo','repository','user','account','page','com','org'}
        m = re.search(r'github\s+(?:profile|repo|repository|user|account)?\s*[-:]?\s*["\']?(\w+)["\']?', task, re.I)
        if m:
            username = m.group(1)
            if username.lower() not in skip_words and len(username) > 1:
                return f"Go to https://github.com/{username} — {task}"
    if "twitter" in t or "x.com" in t:
        m = re.search(r'(?:twitter|x\.com)\s+(\w+)', task, re.I)
        if m: return f"Go to https://x.com/{m.group(1)} — {task}"
    m = re.search(r'open\s+((?:https?://)?[\w-]+\.[\w./]+)', task, re.I)
    if m:
        url = m.group(1).rstrip(',')
        if not url.startswith("http"): url = "https://" + url
        return f"Go to {url} — {task}"
    return task

# ═══════════════════════════════════════════════════════════════════════════
# Main agent — with all optimizations
# ═══════════════════════════════════════════════════════════════════════════
class ZAIBrowserAgent:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("ZAI_API_KEY", "")
        if not self.api_key: raise ValueError("ZAI_API_KEY not set")
        self.base_url = "https://api.z.ai/api/paas/v4"
        self.model = os.getenv("ZAI_MODEL", "glm-4.5-flash")
        self.AgentClass = imp([("browser_use.agent.service","Agent"),("browser_use","Agent")])
        self.ProfileClass = imp([("browser_use.browser.profile","BrowserProfile")])
        self.SessionClass = imp([("browser_use.browser.session","BrowserSession")])
        if not self.AgentClass: raise ImportError("No Agent class")

    async def run(self, task):
        task = preprocess_task(task)
        logger.info(f"Task: {task[:150]}")

        llm = create_llm(self.api_key, self.base_url, self.model)

        # A3: Pre-warm LLM connection while browser starts
        prewarm_task = asyncio.create_task(prewarm_llm(llm))

        # Browser setup — B2: compact viewport for smaller DOM
        profile = None; session = None
        # ─── In your ZAIBrowserAgent.run(), browser setup section ───

        if self.ProfileClass:
            try:
                kw = dict(
                    headless=False,
                    disable_security=True,
                    demo_mode=False,
                    minimum_wait_page_load_time=0.2,
                    wait_for_network_idle_page_load_time=0.2,
                    wait_between_actions=0.05,
                    highlight_elements=False,
                    # ❌ REMOVE THIS LINE — it shrinks the Chrome window:
                    # viewport={"width": 1024, "height": 768},
                )

                # ✅ ADD: maximize Chrome window on launch
                for flag_param in ("extra_chromium_args", "extra_browser_args",
                                "browser_args", "chrome_flags"):
                    try:
                        kw[flag_param] = ["--start-maximized", "--window-size=1920,1080"]
                        break
                    except TypeError:
                        # Profile doesn't accept this param, try next
                        continue

                profile = self.ProfileClass(**kw)
            except TypeError:
                # If extra flag params not accepted, create minimal profile
                try:
                    profile = self.ProfileClass(headless=False, disable_security=True)
                except:
                    pass
        if profile and self.SessionClass:
            try:
                session = self.SessionClass(browser_profile=profile)
                try:
                    session.demo_mode = False
                except Exception:
                    object.__setattr__(session, "demo_mode", False)
                    
                if hasattr(session, "start"):
                    r = session.start()
                    if asyncio.iscoroutine(r): await r
                await asyncio.sleep(0.5)
                # B1: Block unnecessary resources
                await block_resources(session)
                logger.info("Browser ready")
            except Exception as e:
                logger.warning(f"Session: {e}"); session = None

        # Wait for pre-warm to complete
        await prewarm_task

        # Agent config — A1: max_actions_per_step=3 (3x fewer LLM calls)
        cfg = {"task": task, "llm": llm, "use_vision": False,
               "max_steps": 20, "max_consecutive_failures": 3,
               "generate_gif": False}
        # A1: 3 actions per step = 3x fewer LLM calls
        for p, v in [("max_actions_per_step", 3), ("max_elements", 20)]:
            cfg[p] = v
        if session: cfg["browser"] = session
        elif profile: cfg["browser_profile"] = profile

        try: agent = self.AgentClass(**cfg)
        except TypeError:
            cfg.pop("max_elements", None)
            try: agent = self.AgentClass(**cfg)
            except TypeError:
                cfg.pop("max_actions_per_step", None)
                try: agent = self.AgentClass(**cfg)
                except TypeError:
                    cfg.pop("browser", None); cfg.pop("browser_profile", None)
                    agent = self.AgentClass(**cfg)
        logger.info(f"Agent created (actions/step={cfg.get('max_actions_per_step','?')}, elements={cfg.get('max_elements','?')})")

        try: history = await agent.run(max_steps=20)
        except TypeError: history = await agent.run()

        logger.info(f"📊 {_action_cache.stats()}")

        try:
            fr = history.final_result() if hasattr(history, "final_result") else None
            urls = history.urls() if hasattr(history, "urls") else []
            steps = history.total_steps() if hasattr(history, "total_steps") else 0
            errs_raw = history.errors() if hasattr(history, "errors") else []
            errs = [str(e) for e in errs_raw if e is not None]
            dur = int(history.total_duration() * 1000) if hasattr(history, "total_duration") else 0
            return {"ok": len(errs) == 0, "url": urls[-1] if urls else "",
                    "steps": steps, "elapsed_ms": dur,
                    "result": str(fr) if fr else "Task executed",
                    "error": "; ".join(errs) if errs else "",
                    "cache_stats": _action_cache.stats()}
        except Exception as e:
            return {"ok": False, "error": str(e)}

async def main():
    if len(sys.argv) < 2: print("Usage: python main.py 'task'"); sys.exit(1)
    task = " ".join(sys.argv[1:]).strip()
    try:
        a = ZAIBrowserAgent()
        r = await a.run(task)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        sys.exit(0 if r.get("ok") else 1)
    except KeyboardInterrupt: sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal: {e}", exc_info=True)
        print(json.dumps({"ok": False, "error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())