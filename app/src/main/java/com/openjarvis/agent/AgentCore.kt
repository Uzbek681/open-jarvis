package com.openjarvis.agent

import android.content.Context
import android.content.Intent
import android.content.pm.ResolveInfo
import com.openjarvis.accessibility.JarvisAccessibilityService
import com.openjarvis.accessibility.ScreenReader
import com.openjarvis.graphify.AnalysisEngine
import com.openjarvis.graphify.GraphifyRepository
import com.openjarvis.intelligence.AIAppInteractor
import com.openjarvis.intelligence.AIApps
import com.openjarvis.intelligence.AppAnalyzer
import com.openjarvis.intelligence.TaskRouter
import com.openjarvis.intelligence.TaskWorkingMemory
import com.openjarvis.llm.UniversalAdapter
import com.openjarvis.vision.VisionModule
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

class AgentCore(private val context: Context) {

    private val graphifyRepo = GraphifyRepository(context)
    private val analysisEngine = AnalysisEngine(context)
    private val universalAdapter = UniversalAdapter(context)
    private val screenReader = ScreenReader(context)
    private val visionModule = VisionModule.getInstance(context)
    private val taskRouter = TaskRouter(context)
    private val appAnalyzer = AppAnalyzer(context)
    private val aiAppInteractor = AIAppInteractor(context)

    private var workingMemory = TaskWorkingMemory()

    private val scope = CoroutineScope(Dispatchers.IO)
    private val taskMutex = Mutex()

    private val _state = MutableStateFlow<AgentState>(AgentState.Idle)
    val state: StateFlow<AgentState> = _state

    private val systemPrompt = """
You are Open Jarvis — an Android device control AI agent.
The user gives you a cleanCommand in natural language.
You must respond with ONLY a valid JSON array of actions.
No explanation. No markdown fences. No preamble. Pure JSON array only.

AVAILABLE ACTIONS:

open_app → {"action":"open_app","package":"com.package","label":"AppName"}
tap → {"action":"tap","text":"Button text on screen"}
tap_coords → {"action":"tap_coords","x":540,"y":960}
long_press → {"action":"long_press","text":"Element text"}
type → {"action":"type","value":"text to type"}
clear_type → {"action":"clear_type","value":"clears field then types"}
swipe → {"action":"swipe","direction":"up|down|left|right","distance":"short|medium|long"}
scroll → {"action":"scroll","direction":"up|down"}
press_back → {"action":"press_back"}
press_home → {"action":"press_home"}
press_recents → {"action":"press_recents"}
wait_for → {"action":"wait_for","text":"expected text","timeout_ms":3000}
screenshot → {"action":"screenshot"}
read_screen → {"action":"read_screen"}
ai_prompt → {"action":"ai_prompt","package":"com.openai.chatgpt","prompt":"{prompt}","outputKey":"result"}
extract_text → {"action":"extract_text","outputKey":"page_text"}

CURRENT SCREEN CONTENT: {SCREEN_OCR}

APP SELECTION REASONING: {APP_REASONING}

INSTALLED AI APPS: {AI_APPS}

RECENT MEMORY CONTEXT: {GRAPHIFY_CONTEXT}

RULES:
- Always start complex tasks with open_app
- Add wait_for after open_app to confirm app loaded
- If screen content is empty or unclear, add read_screen as first action
- Never assume UI state — always verify with wait_for
- Keep action arrays short: 2-8 steps per task
- Use ai_prompt to delegate complex reasoning to installed AI apps
- If a task is impossible to do safely, return:
  [{"action":"error","message":"reason"}]
""".trimIndent()

    fun executeTask(cleanCommand: String) {
        workingMemory = TaskWorkingMemory()

        scope.launch {
            taskMutex.withLock {
                try {
                    val sanitized = PromptSanitizer.sanitize(cleanCommand)

                    when (sanitized) {
                        is PromptSanitizer.SanitizeResult.Rejected -> {
                            _state.value = AgentState.Error(sanitized.reason)
                            return@withLock
                        }

                        is PromptSanitizer.SanitizeResult.Suspicious -> {
                            _state.value = AgentState.Running("analyzing...")
                        }

                        is PromptSanitizer.SanitizeResult.Clean -> {
                            // Safe command
                        }
                    }

                    val command = when (sanitized) {
                        is PromptSanitizer.SanitizeResult.Suspicious ->
                            sanitized.sanitized

                        is PromptSanitizer.SanitizeResult.Clean ->
                            sanitized.text

                        is PromptSanitizer.SanitizeResult.Rejected ->
                            return@withLock
                    }

                    _state.value = AgentState.Running("analyzing task...")

                    val plan = taskRouter.analyze(command)

                    _state.value = AgentState.Running("reading screen...")

                    val screenText = withContext(Dispatchers.IO) {
                        screenReader.extractAllText()
                    }

                    _state.value = AgentState.Running("getting context...")

                    val memoryContext =
                        graphifyRepo.buildMemoryContext(command)

                    val fullSystem = systemPrompt
                        .replace(
                            "{SCREEN_OCR}",
                            screenText.take(2000)
                        )
                        .replace(
                            "{APP_REASONING}",
                            plan.reasoning
                        )
                        .replace(
                            "{AI_APPS}",
                            getInstalledAIApps()
                        )
                        .replace(
                            "{GRAPHIFY_CONTEXT}",
                            if (memoryContext.isBlank()) {
                                "No recent tasks"
                            } else {
                                memoryContext
                            }
                        )

                    _state.value = AgentState.Running("thinking...")

                    val startTime = System.currentTimeMillis()

                    val result = universalAdapter.complete(
                        fullSystem,
                        command
                    )

                    result.fold(
                        onSuccess = { rawJson ->

                            val latency =
                                System.currentTimeMillis() - startTime

                            val validation =
                                LLMResponseValidator.validate(rawJson)

                            if (
                                !validation.isValid &&
                                validation.errors.isNotEmpty()
                            ) {
                                val errorMessage =
                                    validation.errors.first()

                                _state.value =
                                    AgentState.Error(
                                        "Invalid response: $errorMessage"
                                    )

                                graphifyRepo.logTask(
                                    command,
                                    "failed: validation error",
                                    "",
                                    0
                                )

                                return@fold
                            }

                            var actions =
                                ActionJsonParser.parse(rawJson)

                            if (actions == null) {
                                val retry =
                                    universalAdapter.complete(
                                        fullSystem,
                                        "$command\n\nRespond with JSON array ONLY. No other text."
                                    )

                                actions = retry
                                    .getOrNull()
                                    ?.let {
                                        ActionJsonParser.parse(it)
                                    }
                            }

                            if (actions == null) {
                                _state.value =
                                    AgentState.Error(
                                        "Could not parse AI response"
                                    )

                                graphifyRepo.logTask(
                                    command,
                                    "failed: parse error",
                                    "",
                                    0
                                )

                                return@fold
                            }

                            _state.value =
                                AgentState.Running(
                                    "executing ${actions.size} actions..."
                                )

                            executeActions(actions)

                            graphifyRepo.logTask(
                                cleanCommand = command,
                                result = "success",
                                provider =
                                    universalAdapter.getProviderName(),
                                latencyMs = latency
                            )

                            analysisEngine.analyzeLastTask()

                            _state.value =
                                AgentState.Done(
                                    "done in ${latency}ms"
                                )
                        },

                        onFailure = { error ->

                            val msg = when {
                                error.message?.contains("401") == true ->
                                    "Invalid API key"

                                error.message?.contains("429") == true ->
                                    "Rate limited — wait a moment"

                                error.message?.contains("timeout") == true ->
                                    "Request timed out"

                                error.message?.contains(
                                    "Unable to resolve"
                                ) == true ->
                                    "Network error — check connection"

                                else ->
                                    error.message ?: "Unknown error"
                            }

                            _state.value =
                                AgentState.Error(msg)

                            graphifyRepo.logTask(
                                command,
                                "failed: $msg",
                                "",
                                0
                            )
                        }
                    )

                } catch (e: Exception) {

                    _state.value =
                        AgentState.Error(
                            e.message ?: "Unknown error"
                        )

                    graphifyRepo.logTask(
                        cleanCommand,
                        "failed: ${e.message}",
                        "",
                        0
                    )
                }
            }
        }
    }

    suspend fun testConnection(): Result<Long> {
        return universalAdapter.testConnection()
    }

    fun getCurrentProviderName(): String {
        return universalAdapter.getProviderName()
    }

    fun getStateFlow(): StateFlow<AgentState> {
        return state
    }

    private fun getInstalledAIApps(): String {
        return AIApps.KNOWN_AI_APPS.keys.joinToString(", ")
    }

    suspend fun getAnalyzedAppCount(): Int {
        return appAnalyzer.getAnalyzedCount()
    }

    suspend fun getAIAppCount(): Int {
        return appAnalyzer.getAICount()
    }

    private suspend fun executeActions(actions: List<Action>) {

        for ((index, action) in actions.withIndex()) {

            _state.value =
                AgentState.Running(
                    "action ${index + 1}/${actions.size}"
                )

            when (action.action) {

                Action.OPEN_APP -> {

                    val label = action.label

                    if (label != null) {

                        val packageName =
                            findPackageByLabel(label)

                        if (packageName != null) {

                            Jarvis
