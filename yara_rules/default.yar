rule default_malware_detection
{
    meta:
        description = "Basic rule to detect common malware patterns"
        author = "Security Team"
    
    strings:
        $suspicious_api1 = "VirtualAlloc" nocase
        $suspicious_api2 = "CreateRemoteThread" nocase
        $suspicious_api3 = "WriteProcessMemory" nocase
        $suspicious_api4 = "CreateProcess" nocase
        $suspicious_api5 = "socket" nocase
        $suspicious_string1 = "cmd.exe" nocase
        $suspicious_string2 = "powershell" nocase
        $hex1 = { 55 8B EC } // Common function prologue
        
    condition:
        2 of ($suspicious_api*) or
        any of ($suspicious_string*) or
        $hex1
}