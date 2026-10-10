# toast.ps1 -- 2026-10-09 (Neuro 54). Windows desktop toast from WSL (temp/alert.sh). No modules needed.
param([string]$Title = "neuro", [string]$Msg = "")
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
$x = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$t = $x.GetElementsByTagName("text")
$t.Item(0).AppendChild($x.CreateTextNode($Title)) | Out-Null
$t.Item(1).AppendChild($x.CreateTextNode($Msg)) | Out-Null
$app = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app).Show([Windows.UI.Notifications.ToastNotification]::new($x))
