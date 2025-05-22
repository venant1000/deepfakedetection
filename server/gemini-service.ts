import { GoogleGenerativeAI } from "@google/generative-ai";

// System prompt that guides the model to focus on deepfake-related questions
const SYSTEM_PROMPT = `
You are an expert AI assistant specializing in deepfake detection and digital media forensics.
Your purpose is to educate users about deepfakes, how they work, how to detect them, and their implications.

Guidelines:
1. Only answer questions related to deepfakes, AI-generated media, digital forensics, or media authentication.
2. If a question is unrelated to these topics, politely explain that you're specialized in deepfake education.
3. Provide educational, factual information about deepfake technology.
4. Explain technical concepts in simple terms when possible.
5. Do not provide instructions on how to create malicious deepfakes.
6. Always promote ethical use of AI technology.
7. When uncertain, acknowledge limitations in your knowledge.

Your expertise includes:
- How deepfake technology works
- Methods for detecting deepfakes
- Current state of deepfake technology
- Implications of deepfakes for society
- Tools and techniques for verifying media authenticity
- Legal and ethical considerations around synthetic media
`;

/**
 * Process a user query about deepfakes and generate a response using Gemini AI
 */
export async function processDeepfakeQuery(userMessage: string): Promise<string> {
  try {
    // Get API key from environment
    const apiKey = process.env.GEMINI_API_KEY;
    console.log("Debug - API key: GEMINI_API_KEY exists:", !!process.env.GEMINI_API_KEY);
    
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY is not configured in environment");
    }
    
    // Initialize Gemini API
    const genAI = new GoogleGenerativeAI(apiKey);
    // Use gemini-2.0-flash which is the latest available model
    const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });
    
    // Create a context-aware prompt that includes our guidelines
    const prompt = `
As an expert in deepfake detection and digital media forensics, please answer the following question about deepfakes:

${userMessage}

Remember to focus only on educational information about deepfakes, their detection, or their societal impact. If the question is unrelated to deepfakes, politely redirect the conversation to deepfake topics.
`;

    // Generate content directly instead of using chat
    const result = await model.generateContent(prompt);
    
    const response = result.response;
    return response.text();
  } catch (error) {
    console.error("Error processing chat message with Gemini:", error);
    return "Sorry, I encountered an error processing your question. Please try again later.";
  }
}

/**
 * Get helpful deepfake detection tips from Gemini
 */
export async function getDeepfakeTips(): Promise<string[]> {
  try {
    // Get API key from environment
    const apiKey = process.env.GEMINI_API_KEY;
    console.log("Debug - API key: GEMINI_API_KEY exists:", !!process.env.GEMINI_API_KEY);
    
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY is not configured in environment");
    }
    
    // Initialize Gemini API
    const genAI = new GoogleGenerativeAI(apiKey);
    // Use gemini-2.0-flash which is the latest available model
    const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });
    
    const prompt = "Provide 5 short, practical tips for detecting deepfakes in videos. Make each tip 1-2 sentences only.";
    
    // Generate content directly
    const result = await model.generateContent(prompt);
    
    const response = result.response;
    const text = response.text();
    
    // Split the response into individual tips
    return text
      .split(/\d+\./)
      .map((tip: string) => tip.trim())
      .filter((tip: string) => tip.length > 0);
  } catch (error) {
    console.error("Error generating deepfake tips with Gemini:", error);
    return [
      "Look for unnatural eye movements or blinking patterns.",
      "Check for inconsistencies in facial features like ears or hair.",
      "Pay attention to strange lighting or shadows on the face.",
      "Watch for audio that seems out of sync with lip movements.",
      "Notice unusual skin texture or color changes."
    ];
  }
}

/**
 * Analyze a timeline marker with detailed explanation based on its severity
 * @param markerType Type of anomaly: 'normal', 'warning', or 'danger'
 * @param markerTooltip Brief description of the anomaly
 * @param timestamp When in the video the anomaly occurs
 * @returns Detailed AI analysis of the anomaly
 */
export async function analyzeTimelineMarker(
  markerType: 'normal' | 'warning' | 'danger', 
  markerTooltip: string,
  timestamp: string
): Promise<string> {
  try {
    // Get API key from environment
    const apiKey = process.env.GEMINI_API_KEY;
    
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY is not configured in environment");
    }
    
    // Initialize Gemini API
    const genAI = new GoogleGenerativeAI(apiKey);
    const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });
    
    // Classify the severity level for better context
    const severityLevel = 
      markerType === 'danger' ? 'high' :
      markerType === 'warning' ? 'medium' : 'low';
    
    // Create a specialized prompt for the timeline marker analysis
    const prompt = `
    As a deepfake detection expert, analyze the following anomaly detected in a video at timestamp ${timestamp}:
    
    Anomaly: "${markerTooltip}"
    Severity: ${severityLevel} (${markerType})
    
    Provide a 2-3 sentence detailed explanation of:
    1. What this specific anomaly likely indicates
    2. How this type of manipulation is typically created
    3. Why this is classified as ${severityLevel} risk
    
    Keep your response concise, technical but understandable, and focus on educating about this specific type of deepfake anomaly.
    `;
    
    // Generate the analysis
    const result = await model.generateContent(prompt);
    const response = result.response;
    return response.text().trim();
    
  } catch (error) {
    console.error("Error analyzing timeline marker with Gemini:", error);
    
    // Fallback responses based on severity
    if (markerType === 'danger') {
      return `This is a high-risk anomaly showing ${markerTooltip.toLowerCase()}. This type of manipulation typically involves advanced AI techniques that alter facial features or expressions using specialized deepfake algorithms. The high risk classification indicates this anomaly is highly unlikely to occur naturally and strongly suggests intentional manipulation.`;
    } else if (markerType === 'warning') {
      return `This medium-risk anomaly shows ${markerTooltip.toLowerCase()}. These inconsistencies often result from imperfect frame transitions or expression blending in the deepfake generation process. While concerning, there's a small possibility this could have natural causes in some lighting or camera conditions.`;
    } else {
      return `This low-risk anomaly indicating ${markerTooltip.toLowerCase()} falls within the range of normal video artifacts. While it triggered our detection system, these patterns can often appear in compressed videos or challenging lighting conditions. We've flagged it for transparency but it likely does not indicate manipulation.`;
    }
  }
}