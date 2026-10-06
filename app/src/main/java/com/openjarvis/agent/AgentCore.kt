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
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

class AgentCore(private val context: Context) {

    private val graphifyRepo = GraphifyRepository(context)
    private val analysisEngine = AnalysisEngine(context)
    private val universalAdapter = UniversalAdapter(context)
    private val screenReader = ScreenReader(context)
    private val taskRouter = TaskRouter(context)
    private val appAnalyzer = AppAnalyzer(context)
    private val aiAppInteractor = AIAppInteractor(context)

    private var workingMemory = TaskWorkingMemory()

    private val scope = CoroutineScope(Dispatchers.IO)
    private val taskMutex = Mutex()

    private val _state = MutableStateFlow<AgentState>(AgentState.Idle)
    val state: StateFlow<AgentState> = _state

    private val systemPrompt = """
You are Open Jarvis, an Android device control AI agent.

The user gives you a command in natural language.
Respond ONLY with a valid JSON array of actions.
No explanation. No markdown. No extra text.

AVAILABLE ACTIONS:

open_app:
{"action":"open_app","package":"com.package","label":"AppName"}

tap:
{"action":"tap","text":"Button text"}

tap_coords:
{"action":"tap_coords","x":540,"y":960}

long_press:
{"action":"long_press","text":"Element text"}

type:
{"action":"type","value":"text"}

clear_type:
{"action":"clear_type","value":"text"}

swipe:
{"action":"swipe","direction":"up|down|left|right","distance":"short|medium|long"}

scroll:
{"action":"scroll","direction":"up|down"}

press_back:
{"action":"press_back"}

press_home:
{"action":"press_home"}

press_recents:
{"action":"press_recents"}

wait_for:
{"action":"wait_for","text":"expected text","timeout_ms":3000}

screenshot:
{"action":"screenshot"}

read_screen:
{"action":"read_screen"}

ai_prompt:
{"action":"ai_prompt","package":"com.openai.chatgpt","prompt":"prompt","outputKey":"result"}

extract_text:
{"action":"extract_text","outputKey":"page_text"}

CURRENT SCREEN:
{SCREEN_OCR}

APP REASONING:
{APP_REASONING}

INSTALLED AI APPS:
{AI_APPS}

RECENT MEMORY:
{GRAPHIFY_CONTEXT}

RULES:
- Start complex tasks with open_app.
- Verify the application after opening it.
- If the screen is unclear, use read_screen.
- Never assume the UI state.
- Keep action arrays short.
- Use ai_prompt when another AI app can help.
- If the task cannot be performed safely, return:
[{"action":"error","message":"reason"}]
""".trimIndent()

    fun executeTask(cleanCommand: String) {

        workingMemory = TaskWorkingMemory()

        scope.launch {

            taskMutex.withLock {

                try {

                    val sanitized =
                        PromptSanitizer.sanitize(cleanCommand)

                    if (sanitized is PromptSanitizer.SanitizeResult.Rejected) {
                        _state.value =
                            AgentState.Error(sanitized.reason)
                        return@withLock
                    }

                    if (sanitized is PromptSanitizer.SanitizeResult.Suspicious) {
                        _state.value =
                            AgentState.Running("analyzing...")
                    }

                    val command = when (sanitized) {

                        is PromptSanitizer.SanitizeResult.Clean ->
                            sanitized.text

                        is PromptSanitizer.SanitizeResult.Suspicious ->
                            sanitized.sanitized

                        is PromptSanitizer.SanitizeResult.Rejected ->
                            return@withLock
                    }

                    _state.value =
                        AgentState.Running("analyzing task...")

                    val plan =
                        taskRouter.analyze(command)

                    _state.value =
                        AgentState.Running("reading screen...")

                    val screenText =
                        withContext(Dispatchers.IO) {
                            screenReader.extractAllText()
                        }

                    _state.value =
                        AgentState.Running("getting context...")

                    val memoryContext =
                        graphifyRepo.buildMemoryContext(command)

                    val fullSystem =
                        systemPrompt
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

                    _state.value =
                        AgentState.Running("thinking...")

                    val startTime =
                        System.currentTimeMillis()

                    val result =
                        universalAdapter.complete(
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

                                val error =
                                    validation.errors.first()

                                _state.value =
                                    AgentState.Error(
                                        "Invalid response: $error"
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
                                        "$command\n\nRespond with JSON array ONLY."
                                    )

                                actions =
                                    retry
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

                            val message = when {

                                error.message?.contains("401") == true ->
                                    "Invalid API key"

                                error.message?.contains("429") == true ->
                                    "Rate limited"

                                error.message?.contains("timeout") == true ->
                                    "Request timed out"

                                error.message?.contains(
                                    "Unable to resolve"
                                ) == true ->
                                    "Network error"

                                else ->
                                    error.message ?: "Unknown error"
                            }

                            _state.value =
                                AgentState.Error(message)

                            graphifyRepo.logTask(
                                command,
                                "failed: $message",
                                "",
                                0
                            )
                        }
                    )

                } catch (e: Exception) {

                    val message =
                        e.message ?: "Unknown error"

                    _state.value =
                        AgentState.Error(message)

                    graphifyRepo.logTask(
                        cleanCommand,
                        "failed: $message",
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
