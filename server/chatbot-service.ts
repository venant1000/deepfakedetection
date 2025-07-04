import { GoogleGenerativeAI } from '@google/generative-ai';

// Initialize the Google Generative AI with the API key
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || '');

// Get the Gemini 2.0 Flash model which is the latest available
const model = genAI.getGenerativeModel({ model: 'gemini-2.0-flash' });

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
 * @param userMessage The user's message/question
 * @returns AI-generated response focused on deepfake-related topics
 */
export async function processChatMessage(userMessage: string): Promise<string> {
  try {
    // Create a chat session
    const chat = model.startChat({
      generationConfig: {
        temperature: 0.7,
        maxOutputTokens: 800,
      },
      systemInstruction: SYSTEM_PROMPT,
    });

    // Generate a response from the model
    const result = await chat.sendMessage(userMessage);
    const response = result.response;
    
    // Get the text response
    return response.text();
  } catch (error) {
    console.error('Error processing chat message with Gemini:', error);
    return 'Sorry, I encountered an error processing your question. Please try again later.';
  }
}

/**
 * Get helpful deepfake detection tips from Gemini
 * @returns A list of tips for detecting deepfakes
 */
export async function getDeepfakeTips(): Promise<string[]> {
  try {
    const prompt = 'Provide 5 short, practical tips for detecting deepfakes in videos. Make each tip 1-2 sentences only.';
    
    const result = await model.generateContent(prompt);
    const response = result.response;
    const text = response.text();
    
    // Split the response into individual tips
    return text
      .split(/\d+\./)
      .map(tip => tip.trim())
      .filter(tip => tip.length > 0);
  } catch (error) {
    console.error('Error generating deepfake tips with Gemini:', error);
    return [
      'Look for unnatural eye movements or blinking patterns.',
      'Check for inconsistencies in facial features like ears or hair.',
      'Pay attention to strange lighting or shadows on the face.',
      'Listen for audio that does not sync with lip movements.',
      'Be wary of unusual skin texture or color changes.'
    ];
  }
}