use crate::models::ChatCompletionRequest;

#[derive(Debug, Clone, PartialEq)]
pub enum Intent {
    Agentic,
    Standard,
    Trivial,
}

#[derive(Debug, Clone)]
pub struct IntentClassifier {
    tier3_keywords: Vec<String>,
    tier2_keywords: Vec<String>,
    tier1_keywords: Vec<String>,
    file_count_threshold: usize,
}

impl IntentClassifier {
    pub fn new() -> Self {
        Self {
            tier3_keywords: vec![
                "architect".to_string(),
                "system design".to_string(),
                "multi-file".to_string(),
                "migrate".to_string(),
                "redesign".to_string(),
                "entire codebase".to_string(),
                "refactor the module".to_string(),
                "from scratch".to_string(),
            ],
            tier2_keywords: vec![
                "refactor".to_string(),
                "implement".to_string(),
                "debug".to_string(),
                "optimize".to_string(),
                "explain".to_string(),
                "fix".to_string(),
                "test".to_string(),
                "update".to_string(),
            ],
            tier1_keywords: vec![
                "format".to_string(),
                "simple".to_string(),
                "quick".to_string(),
                "autocomplete".to_string(),
                "complete".to_string(),
                "finish".to_string(),
                "syntax".to_string(),
                "typo".to_string(),
            ],
            file_count_threshold: 10,
        }
    }

    pub fn classify(&self, request: &ChatCompletionRequest) -> Intent {
        let has_system_prompt = request.messages.iter().any(|m| m.role == "system");
        let has_multi_file = self.count_file_references(request) > self.file_count_threshold;
        let has_tools = request.tools.is_some();

        if has_multi_file || has_tools {
            return Intent::Agentic;
        }

        for message in &request.messages {
            let content = message.content.as_text().to_lowercase();

            if self
                .tier3_keywords
                .iter()
                .any(|k| content.contains(&k.to_lowercase()))
            {
                return Intent::Agentic;
            }

            if self
                .tier2_keywords
                .iter()
                .any(|k| content.contains(&k.to_lowercase()))
            {
                return Intent::Standard;
            }

            if self
                .tier1_keywords
                .iter()
                .any(|k| content.contains(&k.to_lowercase()))
            {
                return Intent::Trivial;
            }
        }

        if has_system_prompt {
            Intent::Standard
        } else {
            Intent::Trivial
        }
    }

    fn count_file_references(&self, request: &ChatCompletionRequest) -> usize {
        let mut count = 0;

        for message in &request.messages {
            let content = message.content.as_text();

            count += content.matches("file://").count();
            count += content.matches(".rs").count();
            count += content.matches(".py").count();
            count += content.matches(".js").count();
            count += content.matches(".ts").count();
            count += content.matches(".go").count();
            count += content.matches("src/").count();
            count += content.matches("```").count() / 2;
        }

        count
    }
}

impl Default for IntentClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::MessageContent;

    fn create_test_request(messages: Vec<crate::models::Message>) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "test".to_string(),
            messages,
            max_tokens: None,
            temperature: None,
            top_p: None,
            stream: None,
            tools: None,
            tool_choice: None,
        }
    }

    fn text_message(content: &str) -> crate::models::Message {
        crate::models::Message {
            role: "user".to_string(),
            content: MessageContent::Text(content.to_string()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    #[test]
    fn test_agentic_intent() {
        let classifier = IntentClassifier::new();
        let request =
            create_test_request(vec![text_message("Architect a new system from scratch")]);
        let intent = classifier.classify(&request);
        assert_eq!(intent, Intent::Agentic);
    }

    #[test]
    fn test_standard_intent() {
        let classifier = IntentClassifier::new();
        let request = create_test_request(vec![text_message("Refactor this function")]);
        let intent = classifier.classify(&request);
        assert_eq!(intent, Intent::Standard);
    }

    #[test]
    fn test_trivial_intent() {
        let classifier = IntentClassifier::new();
        let request = create_test_request(vec![text_message("Complete this simple function call")]);
        let intent = classifier.classify(&request);
        assert_eq!(intent, Intent::Trivial);
    }

    #[test]
    fn test_multi_file_detection() {
        let classifier = IntentClassifier::new();
        let request = create_test_request(vec![text_message(
            "file://a.rs file://b.rs file://c.rs file://d.rs file://e.rs file://f.rs file://g.rs file://h.rs file://i.rs file://j.rs file://k.rs",
        )]);
        let intent = classifier.classify(&request);
        assert_eq!(intent, Intent::Agentic);
    }
}
